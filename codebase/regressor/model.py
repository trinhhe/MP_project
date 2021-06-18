from abc import ABC
import torch
from torch import nn
import torch.nn.functional as F
from .body_model import BodyModel
from .LinearModel import LinearModel
from torch import hub
import torchvision.models as models
import numpy as np
import h5py
from .util import batch_rodrigues
import sys


class BaseModel(nn.Module, ABC):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # parameters
        self.bm_path = cfg['data']['bm_path']
        self.in_ch = cfg['model'].get('in_ch', 3)
        self.out_ch = cfg['model'].get('out_ch', 70)
        self.img_resolution = cfg['data']['resy'], cfg['data']['resx']

        self.device = cfg.get('device', 'cuda')
        self.batch_size = cfg['training']['batch_size']

        # body_model
        self.body_model = BodyModel(bm_path=self.bm_path,
                                    num_betas=10,
                                    batch_size=self.batch_size).to(device=self.device)

    @staticmethod
    def create_model(cfg):
        model_name = cfg['model']['name']
        if model_name == 'conv':
            model = ConvModel(cfg)
        elif model_name == 'pre_trained':
            model = ConvModel_Pre(cfg)
        else:
            raise Exception(f'Model `{model_name}` is not defined.')
        return model

    def get_vertices(self, root_loc, root_orient, betas, pose_body, pose_hand):
        """ Fwd pass through the parametric body model to obtain mesh vertices.

        Args:
               root_loc (torch.Tensor): Root location (B, 3).
            root_orient (torch.Tensor): Root orientation (B, 3).
                  betas (torch.Tensor): Shape coefficients (B, 10).
              pose_body (torch.Tensor): Body joint rotations (B, 21*3).
              pose_hand (torch.Tensor): Hand joint rotations (B, 2*3).

        Returns:
            mesh vertices (torch.Tensor): (B, 6890, 3)
        """
        # print(f'batchsize: {self.batch_size}')
        # print(f'rootloc{root_loc.shape}')
        body = self.body_model(trans=root_loc,
                               root_orient=root_orient,
                               pose_body=pose_body,
                               pose_hand=pose_hand,
                               betas=betas)

        vertices = body.v
        return vertices

    def get_joints(self, root_loc, root_orient, betas, pose_body, pose_hand):
        """ Fwd pass through the parametric body model to obtain 3D joint vertices.

        Args:
               root_loc (torch.Tensor): Root location (B, 3).
            root_orient (torch.Tensor): Root orientation (B, 3).
                  betas (torch.Tensor): Shape coefficients (B, 10).
              pose_body (torch.Tensor): Body joint rotations (B, 21*3).
              pose_hand (torch.Tensor): Hand joint rotations (B, 2*3).

        Returns:
            mesh vertices (torch.Tensor): (B, J, 3)
        """
        # print(f'batchsize: {self.batch_size}')
        # print(f'rootloc{root_loc.shape}')
        body = self.body_model(trans=root_loc,
                               root_orient=root_orient,
                               pose_body=pose_body,
                               pose_hand=pose_hand,
                               betas=betas)

        joints = body.Jtr
        # v_a_pose = body.v_a_pose
        # f = body.f
        # abs_bone_transforms = body.abs_bone_transforms
        # bone_transforms = body.bone_transforms
        # betas = body.betas
        # return joints, v_a_pose, f, abs_bone_transforms, bone_transforms, betas
        return joints


class ParameterRegressor(nn.Module):

    def __init__(self, feature_count):
        super().__init__()
        # self.batch_size = batch_size
        # In HMR implementation (https://github.com/MandyMo/pytorch_HMR) and https://github.com/nkolot/SPIN/blob/master/models/hmr.py they initialize with mean_theta (all smpl parameters concatened)
        # init_theta = torch.from_numpy((np.random.random_sample(82) * (0.3+0.3) - 0.3).astype('float32'))
        mean_params = h5py.File(
            "../configs/neutral_smpl_mean_params.h5", mode="r")
        init_theta = np.zeros(82, dtype=np.float)
        init_theta[0:72] = mean_params['pose'][:]
        init_theta[72:] = mean_params['shape'][:]
        self.register_buffer(
            'init_theta', torch.from_numpy(init_theta).float())

        self.fc1 = nn.Linear(feature_count + 82, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.dec_root_orient = nn.Linear(1024, 3)
        self.dec_pos_body = nn.Linear(1024, 63)
        self.dec_pos_hand = nn.Linear(1024, 6)
        self.dec_beta = nn.Linear(1024, 10)

    def forward(self, input, iterations):
        '''
            input: output of nn encoder
        '''
        batch_size = input.shape[0]
        pred_theta = self.init_theta.expand(batch_size, -1)
        pred_root_orient = pred_theta[:, :3]
        pred_pos_body = pred_theta[:, 3:66]
        pred_pos_hand = pred_theta[:, 66:72]
        pred_beta = pred_theta[:, 72:]

        # in HMR they use relu activation functions but not in SPIN where they also make use of the HMR (not sure how this impacts the model)
        for i in range(iterations):
            input_c = torch.cat([input, pred_theta], 1)
            input_c = F.relu(self.fc1(input_c))
            input_c = self.drop1(input_c)
            input_c = F.relu(self.fc2(input_c))
            input_c = self.drop2(input_c)
            pred_root_orient = self.dec_root_orient(input_c) + pred_root_orient
            pred_pos_body = self.dec_pos_body(input_c) + pred_pos_body
            pred_pos_hand = self.dec_pos_hand(input_c) + pred_pos_hand
            pred_beta = self.dec_beta(input_c) + pred_beta

        return pred_root_orient, pred_pos_body, pred_pos_hand, pred_beta


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ConvModel_Pre(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        # self.backbone_f_len = cfg['model'].get('backbone_f_len', 512)
        self._build_net()

    def _build_net(self):
        """ Creates NNs. """

        print(f'Loading resnet_50 model...')
        # hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
        self.backbone = models.resnet50(pretrained=True)

        # train only the classifier layer
        for param in self.backbone.parameters():
            param.requires_grad = True
        # fc_in = self.backbone.fc.in_features
        # fc_out = self.backbone_f_len
        # self.backbone.fc = nn.Linear(in_features=fc_in, out_features=fc_out)

        # no classifier, Iterative regressor takes the output of resnet which is average pooled
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = Identity()
        
        self.regressor = ParameterRegressor(num_ftrs)

    def forward(self, input_data):
        """ Fwd pass.

        Returns (dict):
            mesh vertices (torch.Tensor): (B, 6890, 3)
        """
        image_crop = input_data['image_crop']
        root_loc = input_data['root_loc']
        # print(input_data['file_id'])

        img_encoding = self.backbone(image_crop)
        # img_encoding = img_encoding.view(img_encoding.size(0), -1)
        # print(img_encoding.shape)
        # regress parameters
        iterations = 5
        root_orient, pose_body, pose_hand, betas = self.regressor(
            img_encoding, iterations)

        # regress vertices
        vertices = self.get_vertices(
            root_loc, root_orient, betas, pose_body, pose_hand)

        predictions = {'vertices': vertices,
                       'root_loc': root_loc,
                       'root_orient': root_orient,
                       'betas': betas,
                       'pose_body': pose_body,
                       'pose_hand': pose_hand}

        return predictions


'''
https://github.com/MandyMo/pytorch_HMR/blob/master/src/Discriminator.py

'''


class ShapeDiscriminator(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(fc_layers[-1])
            sys.exit(msg)
        super().__init__(fc_layers, use_dropout, drop_prob, use_ac_func)

    def forward(self, inputs):
        ''' 
        Input: B x 10
        Output: B x 1

        '''
        return self.fc_blocks(inputs)


class PoseDiscriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()

        if channels[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(
                channels[-1])
            sys.exit(msg)

        self.conv_blocks = nn.Sequential()
        l = len(channels)
        for idx in range(l - 2):
            self.conv_blocks.add_module(
                name='conv_{}'.format(idx),
                module=nn.Conv2d(
                    in_channels=channels[idx], out_channels=channels[idx + 1], kernel_size=1, stride=1)
            )

        self.fc_layer = nn.ModuleList()
        for idx in range(23):
            self.fc_layer.append(
                nn.Linear(in_features=channels[l - 2], out_features=1))

    def forward(self, inputs):
        '''
        Input: B x 23 x 9
        Output: B x 1 and B x c x 1 x 23
        '''
        batch_size = inputs.shape[0]
        inputs = inputs.transpose(1, 2).unsqueeze(2)  # to B x 9 x 1 x 23
        internal_outputs = self.conv_blocks(inputs)  # to B x c x 1 x 23
        o = []
        for idx in range(23):
            o.append(self.fc_layer[idx](internal_outputs[:, :, 0, idx]))

        return torch.cat(o, 1), internal_outputs


class FullPoseDiscriminator(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(fc_layers[-1])
            sys.exit(msg)

        super(FullPoseDiscriminator, self).__init__(
            fc_layers, use_dropout, drop_prob, use_ac_func)

    def forward(self, inputs):
        return self.fc_blocks(inputs)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self._create_sub_modules()

    def _create_sub_modules(self):
        '''
            create theta discriminator for 23 joint
        '''

        self.pose_discriminator = PoseDiscriminator([9, 32, 32, 1])
        
        '''
            create full pose discriminator for total 23 joints
        '''
        fc_layers = [23 * 32, 1024, 1024, 1]
        use_dropout = [False, False, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False]
        self.full_pose_discriminator = FullPoseDiscriminator(fc_layers, use_dropout, drop_prob, use_ac_func)

        '''
            shape discriminator for betas
        '''
        fc_layers = [10, 5, 1]
        use_dropout = [False, False]
        drop_prob = [0.5, 0.5]
        use_ac_func = [True, False]
        self.shape_discriminator = ShapeDiscriminator(fc_layers, use_dropout, drop_prob, use_ac_func)

        print('finished create the discriminator modules...')


    def forward(self, thetas):
        '''
        inputs is B x 82(72 + 10)
        output: B x 25 (23 joints + 1 full pose + 1 betas)
        '''
        batch_size = thetas.shape[0]
        # cams, poses, shapes = thetas[:, :3], thetas[:, 3:75], thetas[:, 75:]
        poses, shapes = thetas[:, 0:72], thetas[:, 72:]
        shape_disc_value = self.shape_discriminator(shapes)
        rotate_matrixs = batch_rodrigues(poses.contiguous().view(-1, 3)).view(-1, 24, 9)[:, 1:, :] # We don't consider the root orient
        pose_disc_value, pose_inter_disc_value = self.pose_discriminator(rotate_matrixs)
        full_pose_disc_value = self.full_pose_discriminator(pose_inter_disc_value.contiguous().view(batch_size, -1))
        return torch.cat((pose_disc_value, full_pose_disc_value, shape_disc_value), 1)


class ConvModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.backbone_f_len = cfg['model'].get('backbone_f_len', 500)
        self._build_net()

    def _build_net(self):
        """ Creates NNs. """
        print(f'Loading stock 2d-CNN model...')
        fc_in_ch = 1*(self.img_resolution[0] //
                      2**3)*(self.img_resolution[1]//2**3)

        self.backbone = nn.Sequential(nn.Conv2d(self.in_ch, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.ReLU(),
                                      nn.Conv2d(5, 5, kernel_size=(3, 3), stride=(
                                          2, 2), padding=(1, 1)), nn.ReLU(),
                                      nn.Conv2d(5, 5, kernel_size=(3, 3), stride=(
                                          2, 2), padding=(1, 1)), nn.ReLU(),
                                      nn.Conv2d(5, 1, kernel_size=(3, 3), stride=(
                                          2, 2), padding=(1, 1)), nn.ReLU(),
                                      nn.Flatten(),
                                      nn.Linear(fc_in_ch, self.backbone_f_len))

        self.nn_root_orient = nn.Linear(self.backbone_f_len, 3)
        self.nn_betas = nn.Linear(self.backbone_f_len, 10)
        self.nn_pose_body = nn.Linear(self.backbone_f_len, 63)
        self.nn_pose_hand = nn.Linear(self.backbone_f_len, 6)

    def forward(self, input_data):
        """ Fwd pass.

        Returns (dict):
            mesh vertices (torch.Tensor): (B, 6890, 3)
        """
        image_crop = input_data['image_crop']
        root_loc = input_data['root_loc']
        # print(input_data['file_id'])

        img_encoding = self.backbone(image_crop)

        # regress parameters
        root_orient = self.nn_root_orient(img_encoding)
        betas = self.nn_betas(img_encoding)
        pose_body = self.nn_pose_body(img_encoding)
        pose_hand = self.nn_pose_hand(img_encoding)

        # regress vertices
        vertices = self.get_vertices(
            root_loc, root_orient, betas, pose_body, pose_hand)

        predictions = {'vertices': vertices,
                       'root_loc': root_loc,
                       'root_orient': root_orient,
                       'betas': betas,
                       'pose_body': pose_body,
                       'pose_hand': pose_hand}

        return predictions


if __name__ == '__main__':
    x = torch.rand(10, 500).float()
    # net = ParameterRegressor(x.shape[1])
    # a, b, c, d = net(x, 2)
    # print(a.shape)
    # print(net.named_buffers)
    # mean_params = np.load("../../configs/smpl_mean_params.npz")
    # init_pose = torch.from_numpy(mean_params['pose'][:].astype('float32')).unsqueeze(0)
    # init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
    # init_theta = torch.cat([init_pose, init_shape], 1)
    # print(mean_params['pose'].shape)
    # print(init_pose.shape)
    # print(mean_params.keys())
   
    # mean_values = h5py.File("../../configs/neutral_smpl_mean_params.h5")
   
    # mean = np.zeros(85, dtype = np.float)
    # poses = mean_values['pose']
    # mean[3:75] = poses[:]
    # mean[75:] = mean_values['shape'][:]
    # mean = torch.from_numpy(mean).float()
    # mean = mean.expand(5, -1)
    # print(mean.shape)
    # dis = Discriminator()
    # a = dis(mean)
    # print(a.shape)
    # poses, shapes = mean[:, 3:75], mean[:, 75:]
    # print(poses.shape)
    # print(poses.contiguous().view(-1,3).shape)
    # rotate_matrixs = util.batch_rodrigues(poses.contiguous().view(-1, 3))
    # print(rotate_matrixs.shape)
    # print(rotate_matrixs.view(-1,24,9)[:, 1:, :].shape)
    # print(poses.contiguous().view(-1, 3).shape)
    # rotate_matrixs = util.batch_rodrigues(poses.contiguous().view(-1, 3))
    # print(rotate_matrixs.shape)
    # print(rotate_matrixs.view(-1, 24, 9)[:, 1:, :].shape)

    # pose_disc = PoseDiscriminator([9, 32, 32, 1])
    # print(pose_disc)

    # x = torch.rand(19, 3, 1, 23)
    # a = x.contiguous().view(19, -1)
    # print(a.shape)
