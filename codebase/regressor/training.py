from abc import ABC
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from regressor.util import proj_vertices
from .losses import ReprojectionLoss
# from .losses import chamferLoss


class BaseTrainer(ABC):
    """ Base trainer class. """

    def __init__(self, model, optimizer, vis_dir, cfg):
        self.model = model
        self.optimizer = optimizer
        self.vis_dir = vis_dir
        self.cfg = cfg
        self.reprojection_loss = ReprojectionLoss()
        # self.chamfer_loss = ChamferDistance()

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
        gt_joints = self._compute_gt_joints(data)
        gt_pose = torch.cat([data['root_orient'], data['pose_body'], data['pose_hand']], dim=1)

        pred_vertices = prediction['vertices']
        pred_joints = self._compute_gt_joints(prediction)
        pred_pose = torch.cat([prediction['root_orient'], prediction['pose_body'], prediction['pose_hand']], dim=1)

        vert_diff = gt_vertices - pred_vertices
        joint_diff = gt_joints - pred_joints
        pose_diff = gt_pose - pred_pose
        root_diff = data['root_orient'] - prediction['root_orient']
        betas_diff = data['betas'] - prediction['betas']

        # print(gt_vertices.shape, pred_vertices.shape)
        # print(data.shape, prediction[''].shape)

        loss_dict = {}
        if self.loss_cfg.get('v2v_l1', False):
            loss_dict['v2v_l1'] = torch.abs(vert_diff).mean()
        if self.loss_cfg.get('v2v_l2', False):
            loss_dict['v2v_l2'] = torch.pow(vert_diff, 2).mean()
        if self.loss_cfg.get('j2j_l1', False):
            loss_dict['j2j_l1'] = torch.abs(joint_diff).mean()
        if self.loss_cfg.get('j2j_l2', False):
            loss_dict['j2j_l2'] = torch.pow(joint_diff, 2).mean()
        if self.loss_cfg.get('p2p_l1', False):
            loss_dict['p2p_l1'] = torch.abs(pose_diff).mean()
        if self.loss_cfg.get('p2p_l2', False):
            loss_dict['p2p_l2'] = torch.pow(pose_diff, 2).mean()
        if self.loss_cfg.get('r2r_l2', False):
            loss_dict['r2r_l2'] = torch.pow(root_diff, 2).mean()
        if self.loss_cfg.get('b2b_l2', False):
            loss_dict['b2b_l2'] = torch.pow(betas_diff, 2).mean()
        # if self.loss_cfg.get('chamfer_loss', False):
        #     loss_dict['chamfer_loss'] = chamferLoss(gt_vertices, pred_vertices)
        if self.loss_cfg.get('RL_l1', False):
            loss_dict['RL_l1'] = self.reprojection_loss(self.model, data, prediction)


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
