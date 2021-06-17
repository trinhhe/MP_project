import torch
import torch.nn as nn


class ReprojectionLoss(nn.Module):
    def __init__(self):
        super(ReprojectionLoss, self).__init__()

    def forward(self, model, data, predictions):
        """
        Computing the Reprojection Loss from 3D onto 2D images
        Args:
            inputs: 3D ground truth
            targets: 3D predictions
            smooth: smoothing factor
        Returns:
            2D reprojection Loss
        """
        fx = data['fx']
        fy = data['fy']
        cx = data['cx']
        cy = data['cy']

        @torch.no_grad()
        def get_joints_transformed(bm, batch):
            return bm.get_joints(batch['root_loc'],
                                 batch['root_orient'],
                                 batch['betas'],
                                 batch['pose_body'],
                                 batch['pose_hand'])

        gt_joints_proj = get_joints_transformed(model, data)
        pred_joints_proj = get_joints_transformed(model, predictions)

        # gt_joints_proj = model.get_joints(data['root_loc'],
        #                            data['root_orient'],
        #                            data['pose_body'],
        #                            data['pose_hand'],
        #                            data['betas'])
        #
        # pred_joints_proj = model.get_joints(predictions['root_loc'],
        #                              predictions['root_orient'],
        #                              predictions['pose_body'],
        #                              predictions['pose_hand'],
        #                              predictions['betas'])

        # confer proj_vertices: x_new = x/z * fx + cx and y_new =  y/z * fy + cy
        gt_x = gt_joints_proj[..., 0] / gt_joints_proj[..., 2] * fx.view(-1, 1) + cx.view(-1, 1)
        gt_y = gt_joints_proj[..., 1] / gt_joints_proj[..., 2] * fy.view(-1, 1) + cy.view(-1, 1)

        pred_x = pred_joints_proj[..., 0] / pred_joints_proj[..., 2] * fx.view(-1, 1) + cx.view(-1, 1)
        pred_y = pred_joints_proj[..., 1] / pred_joints_proj[..., 2] * fy.view(-1, 1) + cy.view(-1, 1)
        # print(gt_x, pred_x)
        # print(gt_joints_proj.shape, pred_joints_proj.shape)
        # print(gt_x.shape, gt_y.shape)
        # print(torch.stack([gt_x, gt_y]).shape)
        gt_joints_proj = torch.stack([gt_x,gt_y]).permute(1,2,0)  # 2, bs, x/y -> bs, x, y
        # print(gt_joints_proj.shape)
        pred_joints_proj = torch.stack([pred_x, pred_y]).permute(1, 2, 0)  # 2, bs, x/y -> bs, x, y
        # print(pred_joints_proj.shape)

        # # body_model = model.get_body_model(trans=['root_loc'],
        # #                                   root_orient=data['root_orient'],
        # #                                   pose_body=data['pose_body'],
        # #                                   pose_hand=data['pose_hand'],
        # #                                   betas=data['betas'])
        # @torch.no_grad()
        # def project_2d(self, batch_input_3d):
        #     body = model.get_body_model(trans=batch_input_3d['root_loc'],
        #                                 root_orient=batch_input_3d['root_orient'],
        #                                 pose_body=batch_input_3d['pose_body'],
        #                                 pose_hand=batch_input_3d['pose_hand'],
        #                                 betas=batch_input_3d['betas'])
        #
        #     return body['Jtr']
        proj_joints_diff = gt_joints_proj - pred_joints_proj
        return torch.abs(proj_joints_diff).mean()
