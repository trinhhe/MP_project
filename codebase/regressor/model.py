from abc import ABC
import torch
from torch import nn
from .body_model import BodyModel
from torch import hub
import torchvision.models as models
import numpy as np
import h5py

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

        body = self.body_model(trans=root_loc,
                               root_orient=root_orient,
                               pose_body=pose_body,
                               pose_hand=pose_hand,
                               betas=betas)

        vertices = body.v
        return vertices

    def get_joints(self, root_loc, root_orient, betas, pose_body, pose_hand):
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

        body = self.body_model(trans=root_loc,
                               root_orient=root_orient,
                               pose_body=pose_body,
                               pose_hand=pose_hand,
                               betas=betas)

        joints = body.Jtr
        return joints


class ConvModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.backbone_f_len = cfg['model'].get('backbone_f_len', 500)
        self._build_net()

    def _build_net(self):
        """ Creates NNs. """
        print(f'Loading stock 2d-CNN model...')
        fc_in_ch = 1*(self.img_resolution[0]//2**3)*(self.img_resolution[1]//2**3)

        self.backbone = nn.Sequential(nn.Conv2d(self.in_ch, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.ReLU(),
                                      nn.Conv2d(5, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.ReLU(),
                                      nn.Conv2d(5, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.ReLU(),
                                      nn.Conv2d(5, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.ReLU(),
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
        vertices = self.get_vertices(root_loc, root_orient, betas, pose_body, pose_hand)

        predictions = {'vertices': vertices,
                       'root_loc': root_loc,
                       'root_orient': root_orient,
                       'betas': betas,
                       'pose_body': pose_body,
                       'pose_hand': pose_hand}

        return predictions


class ConvModel_Pre(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.backbone_f_len = cfg['model'].get('backbone_f_len', 500)
        self._build_net()

    def _build_net(self):
        """ Creates NNs. """

        print(f'Loading resnet_18 model...')
        self.backbone = models.resnet18(pretrained=True)

        # train only the classifier layer
        for param in self.backbone.parameters():
            param.requires_grad = True
        fc_in = self.backbone.fc.in_features
        fc_out = self.backbone_f_len
        self.backbone.fc = nn.Linear(in_features=fc_in, out_features=fc_out)

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
        vertices = self.get_vertices(root_loc, root_orient, betas, pose_body, pose_hand)

        predictions = {'vertices': vertices,
                       'root_loc': root_loc,
                       'root_orient': root_orient,
                       'betas': betas,
                       'pose_body': pose_body,
                       'pose_hand': pose_hand}

        return predictions



class ParameterRegressor(nn.Module):

    def __init__(self, feature_count):
        super().__init__()
        # self.batch_size = batch_size
        # In HMR implementation (https://github.com/MandyMo/pytorch_HMR) they initialize with mean_theta (all smpl parameters concatened)
        # Dunno how to get them without downloading their file, so I just get some random numbers between -0.3;0.3 and increase iterations
        # init_theta = torch.from_numpy((np.random.random_sample(82) * (0.3+0.3) - 0.3).astype('float32'))
        mean_params = h5py.File("../configs/neutral_smpl_mean_params.h5", mode="r")
        init_theta = np.zeros(82, dtype=np.float)
        init_theta[0:72] = mean_params['pose'][:]
        init_theta[72:] = mean_params['shape'][:]
        self.register_buffer('init_theta', torch.from_numpy(init_theta).float())

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

        for i in range(iterations):
            input_c = torch.cat([input, pred_theta], 1)
            input_c = self.fc1(input_c)
            input_c = self.drop1(input_c)
            input_c = self.fc2(input_c)
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


class HMR_pre(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        # self.backbone_f_len = cfg['model'].get('backbone_f_len', 512)
        self._build_net()

    def _build_net(self):
        """ Creates NNs. """

        print(f'Loading resnet_50 model...')
        self.backbone = models.resnet50(pretrained=True)  # hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)

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
        root_orient, pose_body, pose_hand, betas = self.regressor(img_encoding, iterations)

        # regress vertices
        vertices = self.get_vertices(root_loc, root_orient, betas, pose_body, pose_hand)

        predictions = {'vertices': vertices,
                       'root_loc': root_loc,
                       'root_orient': root_orient,
                       'betas': betas,
                       'pose_body': pose_body,
                       'pose_hand': pose_hand}

        return predictions