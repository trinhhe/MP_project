from abc import ABC
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from regressor.util import proj_vertices, batch_rodrigues, align_by_pelvis, proj_3D_to_2D
# from util import proj_vertices
class BaseTrainer(ABC):
    """ Base trainer class. """

    def __init__(self, model, optimizer, vis_dir, cfg):
        self.model = model
        self.optimizer = optimizer
        self.vis_dir = vis_dir
        self.cfg = cfg

        self.device = cfg['device']
        self.loss_cfg = cfg['loss']

    def _data2device(self, data, device=None):
        """ Move batch to device.

        Args:
            data (dict): dict of tensors
        """
        if device is None:
            device = self.device

        for key in data.keys():
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(device=device)

    @torch.no_grad()
    def evaluate(self, val_loader):
        """ Performs an evaluation.

        Args:
            val_loader (dict of DataLoader): Dictionary with `fuse`, `render`, and `geometry_render` dataloader.

        Returns:
            eval_dict (dict): Dictionary with evaluation loss values.
        """

        raise NotImplementedError()

    def train_step(self, *args, **kwargs):
        """ Performs a training step. """
        raise NotImplementedError

    def test_step(self, *args, **kwargs):
        """ Performs a training step. """
        raise NotImplementedError


class ConvTrainer(BaseTrainer):
    """ Trainer class. """

    def __init__(self, model, optimizer, vis_dir, cfg):
        super().__init__(model, optimizer, vis_dir, cfg)

    def train_step(self, data):
        """ A single training step.

        Args:
            data (dict): data dictionary

        Returns:
            dict with loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        loss_dict = self.compute_training_loss(data)
        loss_dict['total_loss'].backward()
        self.optimizer.step()

        return {k: v.item() for k, v in loss_dict.items()}

    @torch.no_grad()
    def test_step(self, data, key_list):
        """ A single test step.

        Args:
            data (dict): data dictionary
            key_list (list of str): a list of attributes that indicate which values to return

        Returns (dict):
               root_loc (torch.Tensor):  (B, 3).
            root_orient (torch.Tensor): (B, 3)
                  betas (torch.Tensor): (B, 10)
              pose_body (torch.Tensor): (B, 21*3)
              pose_hand (torch.Tensor): (B, 2*3)
        """
        self.model.eval()
        self._data2device(data)
        prediction = self.model.forward(data)

        return {key: prediction[key].cpu() for key in key_list}

    @torch.no_grad()
    def evaluate(self, val_loader, max_images=5):
        """ Performs an evaluation.

        Args:
            val_loader (dict of DataLoader): Dictionary with `fuse`, `render`, and `geometry_render` dataloader.
            max_images (ing): how many images to generate.

        Returns:
            eval_dict (dict): Dictionary with evaluation loss values.
        """
        self.model.eval()
        eval_list = defaultdict(list)
        eval_images = None

        for data in tqdm(val_loader):
            self._data2device(data)
            prediction = self.model.forward(data)

            ret_images = eval_images is None or eval_images.shape[0] < 10
            eval_step_dict, imgs = self.compute_val_loss(prediction, data, ret_images)

            if eval_images is None:
                eval_images = imgs[:max_images, ...].cpu()
            elif eval_images.shape[0] < max_images:
                imgs = eval_images[:max_images - eval_images.shape[0], ...].cpu()
                eval_images = torch.cat((eval_images, imgs), dim=0)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: float(np.mean(v)) for k, v in eval_list.items()}

        # concatenate images along the W dimension
        eval_images = eval_images.permute(2, 0, 3, 1).contiguous()
        eval_images = eval_images.view(eval_images.shape[0], -1, 3).permute(2, 0, 1)

        return eval_dict, eval_images


    def compute_training_loss(self, data):
        """ Computes loss values.

        Notes: Training loss is not correct/reliable for TSDF Fusion.

        Returns:
            dict of torch loss objects.
        """
        self._data2device(data)
        prediction = self.model.forward(data)

        gt_vertices = self._compute_gt_vertices(data)
        pred_vertices = prediction['vertices']

        vert_diff = gt_vertices - pred_vertices
        loss_dict = {}
        if self.loss_cfg.get('v2v_l1', False):
            loss_dict['v2v_l1'] = torch.abs(vert_diff).mean()
        if self.loss_cfg.get('v2v_l2', False):
            loss_dict['v2v_l2'] = torch.pow(vert_diff, 2).mean()

        loss_dict['total_loss'] = sum(self.loss_cfg.get(f'{key}_w', 1.) * val for key, val in loss_dict.items())

        return loss_dict

    @torch.no_grad()
    def _compute_gt_vertices(self, data):
        return self.model.get_vertices(data['root_loc'],
                                       data['root_orient'],
                                       data['betas'],
                                       data['pose_body'],
                                       data['pose_hand'])

    @torch.no_grad()
    def _compute_gt_joints(self, data):
        return self.model.get_joints(data['root_loc'],
                                       data['root_orient'],
                                       data['betas'],
                                       data['pose_body'],
                                       data['pose_hand'])


    def compute_val_loss(self, prediction, data, ret_images=False):
        gt_vertices = self._compute_gt_vertices(data)
        pred_vertices = prediction['vertices']

        images = None
        if ret_images:
            gt_images = proj_vertices(gt_vertices, data['image'], data['fx'], data['fy'], data['cx'], data['cy'])
            pred_images = proj_vertices(pred_vertices, data['image'], data['fx'], data['fy'], data['cx'], data['cy'])
            images = torch.cat((gt_images, pred_images), dim=2)

        loss_dict = {
            'v2v_l2': torch.pow(gt_vertices - pred_vertices, 2).mean().item()
        }
        return loss_dict, images


class HMRTrainer(ABC):
    """ Base trainer class. """

    def __init__(self, generator, discriminator, gen_optimizer, disc_optimizer, vis_dir, cfg):
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.vis_dir = vis_dir
        self.cfg = cfg

        self.device = cfg['device']
        self.loss_cfg = cfg['loss']

    def _data2device(self, data, device=None):
        """ Move batch to device.

        Args:
            data (dict): dict of tensors
        """
        if device is None:
            device = self.device

        for key in data.keys():
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(device=device)

    def train_step(self, data):
        """ A single training step.

        Args:
            data (dict): data dictionary

        Returns:
            dict with loss values
        """
        self.generator.train()
        self.discriminator.train()
        self.gen_optimizer.zero_grad()
        self.disc_optimizer.zero_grad()
        loss_dict = self.compute_training_loss(data)
        
        loss_dict['gen_loss'].backward()
        self.gen_optimizer.step()
        loss_dict['disc_loss'].backward()
        self.disc_optimizer.step()

        return {k: v.item() for k, v in loss_dict.items()}


    @torch.no_grad()
    def test_step(self, data, key_list):
        """ A single test step.

        Args:
            data (dict): data dictionary
            key_list (list of str): a list of attributes that indicate which values to return

        Returns (dict):
               root_loc (torch.Tensor):  (B, 3).
            root_orient (torch.Tensor): (B, 3)
                  betas (torch.Tensor): (B, 10)
              pose_body (torch.Tensor): (B, 21*3)
              pose_hand (torch.Tensor): (B, 2*3)
        """
        self.generator.eval()
        self._data2device(data)
        prediction = self.generator.forward(data)

        return {key: prediction[key].cpu() for key in key_list}

    @torch.no_grad()
    def evaluate(self, val_loader, max_images=5):
        """ Performs an evaluation.

        Args:
            val_loader (dict of DataLoader): Dictionary with `fuse`, `render`, and `geometry_render` dataloader.
            max_images (ing): how many images to generate.

        Returns:
            eval_dict (dict): Dictionary with evaluation loss values.
        """
        self.generator.eval()
        eval_list = defaultdict(list)
        eval_images = None

        for data in tqdm(val_loader):
            self._data2device(data)
            prediction = self.generator.forward(data)

            ret_images = eval_images is None or eval_images.shape[0] < 10
            eval_step_dict, imgs = self.compute_val_loss(prediction, data, ret_images)

            if eval_images is None:
                eval_images = imgs[:max_images, ...].cpu()
            elif eval_images.shape[0] < max_images:
                imgs = eval_images[:max_images - eval_images.shape[0], ...].cpu()
                eval_images = torch.cat((eval_images, imgs), dim=0)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: float(np.mean(v)) for k, v in eval_list.items()}

        # concatenate images along the W dimension
        eval_images = eval_images.permute(2, 0, 3, 1).contiguous()
        eval_images = eval_images.view(eval_images.shape[0], -1, 3).permute(2, 0, 1)

        return eval_dict, eval_images


    def compute_training_loss(self, data):
        """ Computes loss values.

        Notes: Training loss is not correct/reliable for TSDF Fusion.

        Returns:
            dict of torch loss objects.
        """
        self._data2device(data)
        prediction = self.generator.forward(data)

        gt_vertices, gt_kp_3d= self._compute_compute_gt_vertices_and_joints(data)
        ####TODO gt_kp_2d with points_dict['center_img'] and scale from dataset.py to project kp3d to kp2d
        gt_betas = data['betas']
        gt_poses = torch.cat([data['root_orient'], data['pose_body'], data['pose_hand'] ], 1)
        gt_thetas = torch.cat([gt_poses, gt_betas], 1)
        pred_vertices = prediction['vertices']
        pred_kp_3d = prediction['joints']
        pred_betas = prediction['betas']
        pred_poses = torch.cat([prediction['root_orient'], prediction['pose_body'],prediction['pose_hand'] ], 1)
        pred_thetas = torch.cat([pred_poses, pred_betas], 1)
        pred_kp_2d = proj_3D_to_2D(pred_kp_3d, data['fx'], data['fy'], data['cx'], data['cy'])
        gt_kp_2d = proj_3D_to_2D(gt_kp_3d, data['fx'], data['fy'], data['cx'], data['cy'])
        
        vert_diff = gt_vertices - pred_vertices
        loss_dict = {}
        if self.loss_cfg.get('v2v_l1', False):
            loss_dict['v2v_l1'] = torch.abs(vert_diff).mean()
        if self.loss_cfg.get('v2v_l2', False):
            loss_dict['v2v_l2'] = torch.pow(vert_diff, 2).mean()
        if self.loss_cfg.get('kp_2d_l1', False):
            loss_dict['kp_2d_l1'] = self.kp_2d_l1_loss(gt_kp_2d, pred_kp_2d)
        if self.loss_cfg.get('kp_3d_l2', False):
            loss_dict['kp_3d_l2'] = self.kp_3d_l2_loss(gt_kp_3d, pred_kp_3d)
        if self.loss_cfg.get('shape_l2', False):
            loss_dict['shape_l2'] = self.shape_l2_loss(gt_betas, pred_betas)
        if self.loss_cfg.get('pose_l2', False):
            loss_dict['pose_l2'] = self.pose_l2_loss(gt_poses, pred_poses)
        if self.loss_cfg.get('gen_disc_l2', False):
            loss_dict['gen_disc_l2'] = self.gen_disc_l2_loss(pred_thetas)

        loss_dict['gen_loss'] = sum(self.loss_cfg.get(f'{key}_w', 1.) * val for key, val in loss_dict.items())

        fake_disc_value, real_disc_value = self.discriminator(pred_thetas.detach()), self.discriminator(gt_thetas) # detach to only compute gradients for dscriminator

        if self.loss_cfg.get('disc_loss', False):
            d_disc_real, d_disc_predict,  d_disc_loss =  self.adv_disc_l2_loss(real_disc_value, fake_disc_value)
            loss_dict['disc_loss'] = d_disc_loss * self.loss_cfg.get('disc_loss_w', 1.)
            loss_dict['d_disc_real'] =  d_disc_real * self.loss_cfg.get('disc_loss_w', 1.)
            loss_dict['d_disc_predict'] = d_disc_predict * self.loss_cfg.get('disc_loss_w', 1.)


        return loss_dict

    def kp_2d_l1_loss(self, gt_kp_2d, pred_kp_2d):
        '''
            Inputs: B x K x 2
            Ouputs: L1 loss
        '''
        return torch.pow(gt_kp_2d - pred_kp_2d, 2).mean()

    def kp_3d_l2_loss(self, gt_kp_3d, pred_kp_3d):
        '''
            Inputs: B x K x 3
            Ouputs: L2 loss
        '''
        ### align by pelvis first
        gt_kp_3d = align_by_pelvis(gt_kp_3d) 
        pred_kp_3d = align_by_pelvis(pred_kp_3d)
        return torch.pow(gt_kp_3d - pred_kp_3d, 2).mean()

    def shape_l2_loss(self, gt_shape, pred_shape):
        """
            Inputs: B x 10
        """
        return torch.pow(gt_shape - pred_shape, 2).mean()

    def pose_l2_loss(self, gt_pose, pred_pose):
        """
            Inputs: B x 6890 x 3
        """
        batch_size = gt_pose.shape[0]
        gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view([batch_size, -1, 3, 3])
        pred_rotmat = batch_rodrigues(pred_pose.view(-1, 3)).view([batch_size, -1, 3, 3])
        return torch.pow(gt_rotmat - pred_rotmat, 2).mean()

    def gen_disc_l2_loss(self, pred_thetas):
        """
            Inputs: B x 82 
        """
        disc_value = self.discriminator(pred_thetas)
        return torch.pow(disc_value - 1, 2).mean()

    def adv_disc_l2_loss(self, real_disc_value, fake_disc_value):
        ka = real_disc_value.shape[0]
        kb = fake_disc_value.shape[0]
        lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
        return la, lb, la + lb

    @torch.no_grad()
    def _compute_gt_vertices(self, data):
        return self.generator.get_vertices(data['root_loc'],
                                       data['root_orient'],
                                       data['betas'],
                                       data['pose_body'],
                                       data['pose_hand'])

    @torch.no_grad()
    def _compute_gt_joints(self, data):
        return self.generator.get_joints(data['root_loc'],
                                       data['root_orient'],
                                       data['betas'],
                                       data['pose_body'],
                                       data['pose_hand'])


    @torch.no_grad()
    def _compute_compute_gt_vertices_and_joints(self, data):
        return self.generator.get_vertices_and_joints(data['root_loc'],
                                       data['root_orient'],
                                       data['betas'],
                                       data['pose_body'],
                                       data['pose_hand'])

    def compute_val_loss(self, prediction, data, ret_images=False):
        gt_vertices = self._compute_gt_vertices(data)
        pred_vertices = prediction['vertices']

        images = None
        if ret_images:
            gt_images = proj_vertices(gt_vertices, data['image'], data['fx'], data['fy'], data['cx'], data['cy'])
            pred_images = proj_vertices(pred_vertices, data['image'], data['fx'], data['fy'], data['cx'], data['cy'])
            images = torch.cat((gt_images, pred_images), dim=2)

        loss_dict = {
            'v2v_l2': torch.pow(gt_vertices - pred_vertices, 2).mean().item()
        }
        return loss_dict, images


if __name__ == '__main__':
    import os
    cwd = os.getcwd()
    print(cwd)
    # import sys
    # sys.path.append()
    
    # parser = argparse.ArgumentParser(description='Train pipeline.')
    # parser.add_argument('--config', type=str, default='../configs/default.yaml',  help='Path to a config file.')
    # _args = parser.parse_args()
    # print(_args)

    # x = torch.rand(10, 500).float()
    # disc = Discriminator()
    # gen = ConvModel_Pre(config.load_config(_args))
