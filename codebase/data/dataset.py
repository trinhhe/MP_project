import os
from glob import glob
from os.path import join, exists, basename, splitext

import cv2
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt

from data.macros import cam_params
from data.util import crop, pose_processing, augm_params, rgb_add_noise, crop_new


class H36MDataset(torch.utils.data.Dataset):
    """ H36M dataset class. """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, dataset_folder, img_folder, subjects, mode, img_size):
        """ Initialization of the the H36M dataset.

        Args:
            dataset_folder (str): dataset folder
                img_folder (str): dataset folder
                 subjects (list): list of subjects to use
                      mode (str): train, val, or test mode
                img_size (tuple): image resolution. Tuple of integers (height, width).
        """
        # Attributes
        self.dataset_folder = dataset_folder
        self.img_folder = img_folder
        self.subjects = subjects
        self.mode = mode
        self.img_size = img_size

        self.data = self._prepare_files()

    def _prepare_files(self):
        """ List data files. """
        data = []
        for subject in self.subjects:
            subject_dir = join(self.dataset_folder, subject)
            action_dirs = glob(join(subject_dir, '*'))
            actions = set()
            for action_dir in action_dirs:
                actions.add(basename(action_dir).split('.')[0])

            actions = sorted(list(actions))

            for action in actions:
                for cam_idx, cam in enumerate(cam_params):
                    img_dir = join(self.img_folder, subject, 'Images', action + '.' + cam['id'])
                    points_dir = join(subject_dir, action + '.' + cam['id'])

                    points_files = sorted(glob(join(points_dir, '*.npz')))
                    # for f_idx, points_file in enumerate(points_files):

                    for points_file in points_files:
                        data_frame = int(splitext(basename(points_file))[0])
                        img_frame_id = data_frame + 1

                        data.append({    'subject': subject,
                                        'sequence': action,
                                         'cam_idx': cam_idx,
                                        'img_file': join(img_dir, f'{img_frame_id:06d}.jpg'),
                                     'points_file': points_file})
        return data

    def __len__(self):
        """ Returns the length of the dataset. """

        return len(self.data)

    def _get_file_id(self, idx):
        f_path = self.data[idx]['points_file']
        f_path = '/'.join(f_path.split(os.path.sep)[-3:])
        return f_path

    def __getitem__(self, idx):
        """ Returns an item of the dataset.

        Args:
            idx (int): data ID.
        """
        data_path = self.data[idx]['points_file']
        img_path = self.data[idx]['img_file']
        cam_idx = self.data[idx]['cam_idx']
        cv2.setNumThreads(0)

        # Load img files
        assert exists(img_path)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # Load joints & keypoints (root_loc, center_img, scale_img, root_orient, betas, pose_body, pose_hand)
        points_dict = dict(np.load(data_path))
        # points_dict.files
        # for keys, items in points_dict.items():
        #    print(keys, items)

        # TODO: Hint: implement data augmentation
        # Crop image according to the supplied bounding box

        crop_ver = 1
        ###############
        # Crop original
        ###############
        if crop_ver == 0:
            image_crop = crop(image, points_dict['center_img'], points_dict['scale_img'], self.img_size).astype(np.float32)
            image_crop = (image_crop / 255.0 - self.mean) / self.std  # normalize
            # print(f'Min/max/mean: {image_crop.min()}, {image_crop.max()}, {image_crop.mean()}')
            # plt.imshow(image_crop);plt.show()

        ##########
        # Crop new
        ##########
        if crop_ver == 1:
            # Take augmentation params (if mode = training)
            flip, pn, rot, sc = augm_params(self)
            # print(f'Flip/pn/rot/scale: {flip}, {pn}, {rot}, {sc}')

            # 0.) Apply img augmentation (random; scale, rot, flip, pixel_noise)
            image_crop, t_no_scale, h_rot = crop_new(image, points_dict['center_img'], sc * points_dict['scale_img'], self.img_size, rot=rot, flip=flip)
            # image_crop = (image_crop / 255.0 - self.mean) / self.std     # normalize
            # image_crop = np.clip(image_crop, a_min=0., a_max=1.)       # clip
            # print(f'Min/max/mean: {image_crop.min()}, {image_crop.max()}, {image_crop.mean()}')
            # plt.imshow(image_crop);plt.show()

            # 1b.) Add pixel noise in a channel-wise manner
            # Note: seems to be this method has different effect on the final img (more color change, than Gaussian noise)
            image_crop = rgb_add_noise(image_crop, pn)
            image_crop = (image_crop / 255.0 - self.mean) / self.std
            # print(f'Min/max/mean: {image_crop.min()}, {image_crop.max()}, {image_crop.mean()}')
            # plt.imshow(image_crop);plt.show()

            # 3.) Apply Pose augmentation (rot, flip)
            # rot=0
            full_pose = np.concatenate((points_dict['root_orient'], points_dict['pose_body'], points_dict['pose_hand']))
            points_dict['root_orient'], points_dict['pose_body'], points_dict['pose_hand'] = pose_processing(full_pose, r=rot, f=flip)

            # 4.) Add random Gaussian noise to pose
            # mu, sigma = 0, 0.1  # set mean and standard deviation
            # noise_factor = 1
            # gaussian_noise = np.random.normal(mu, sigma, size=points_dict['pose_body'].shape)
            # points_dict['pose_body'] = points_dict['pose_body'] + noise_factor * gaussian_noise.astype('float32')
            # gaussian_noise = np.random.normal(mu, sigma, size=points_dict['pose_hand'].shape)
            # points_dict['pose_hand'] = points_dict['pose_hand'] + noise_factor * gaussian_noise.astype('float32')



        if crop_ver == 1:
            # 1.) Add random Gaussian noise to img
            mu, sigma = 0, 0.1  # set mean and standard deviation
            noise_factor = 0.5  # 3
            gaussian_noise = np.random.normal(mu, sigma, size=image_crop.shape)
            image_crop = image_crop + noise_factor * gaussian_noise.astype('float32')
            # # image_crop = np.clip(image_crop, a_min=0., a_max=1.)  # clip
            # # plt.imshow(image_crop);plt.show()
            #
            # 2.) Add random flip
            # rnd_flip = np.random.choice([0, 1], size=1)
            # print(rnd_flip)

            # if rnd_flip == True:
            #     # Image
            #     image_crop = np.flip(image_crop, axis=1).copy()
            #     # image_crop = (image_crop / 255.0 - self.mean) / self.std  # normalization
            #     # image_crop = np.clip(image_crop, a_min=0., a_max=1.)  # clip



        # -> PyTorch tensor format
        image_crop = image_crop.transpose([2, 0, 1])


        data_out = {'image_crop': image_crop,
                      'root_loc': points_dict['root_loc'],
                            'cx': np.array([cam_params[cam_idx]['center'][0]]),
                            'cy': np.array([cam_params[cam_idx]['center'][1]]),
                            'fx': np.array([cam_params[cam_idx]['focal_length'][0]]),
                            'fy': np.array([cam_params[cam_idx]['focal_length'][1]]),
                           'idx': idx,
                       'file_id': self._get_file_id(idx)}

        if self.mode in ['train', 'val']:
                                            # points_dict['root_orient']points_dict['pose_body']points_dict['pose_hand']
            data_out.update({     'betas': points_dict['betas'],
                              'pose_body': points_dict['pose_body'],
                              'pose_hand': points_dict['pose_hand'],
                            'root_orient': points_dict['root_orient']
                            })

        if self.mode in ['val']:
            if image.shape[0] == 1000:
                # pad 1 pixel to nicely fit in memory; the full image is used only for debugging/visualization
                image = np.pad(image, pad_width=((0, 2), (0, 0), (0, 0)), mode='reflect')

            data_out.update({'image': image.transpose([2, 0, 1])})    # -> PyTorch tensor format

        # float64 -> float32
        data_out.update({
            key: val.astype(np.float32)
            for key, val in data_out.items() if isinstance(val, np.ndarray) and val.dtype == np.float64
        })

        return data_out

    @staticmethod
    def collate_fn(data):
        """ A custom collate function to handle file names. """
        batch = []
        for sample in data:
            batch.append({key: val for key, val in sample.items() if key != 'file_id'})

        batch = default_collate(batch)
        batch['file_id'] = [x['file_id'] for x in data]

        return batch
