import argparse
import config
import torch
from regressor.util import proj_vertices
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



def proj_verts(points, images, fx, fy, cx, cy):
    """ Projects 3D points onto image plane.

    Args:
        points (torch.Tensor): 3D points (B, N, 3)
                       images:  3D points (B, 3, H, W)
                           fx: focal length (B,)
                           fy: focal length (B,)
                           cx: focal length (B,)
                           cy: focal length (B,)

    Returns:
        rendered_img (torch.Tensor): Images with marked vertex locations (B, 3, H, W).
    """
    B, H, W = images.shape[0], images.shape[2], images.shape[3]
    device = images.device

    # compute pixel locations
    x = points[..., 0] / points[..., 2] * fx.view(-1, 1) + cx.view(-1, 1)
    y = points[..., 1] / points[..., 2] * fy.view(-1, 1) + cy.view(-1, 1)

    # round pixel locations
    x = (x+0.5).int().long()
    y = (y+0.5).int().long()

    plt.figure()
    plt.scatter(x[1].cpu().numpy(),y[1].cpu().numpy())

    # discard invalid pixels
    valid = ~((x < 0) | (y < 0) | (x >= W) | (y >= H))
    idx = (W * y + x) + W * H * torch.arange(0, B, device=device, dtype=torch.long).view(B, 1)
    idx = idx.view(-1)[valid.view(-1)]

    # mark selected pixels
    rendered_img = torch.clone(images).permute(0, 2, 3, 1)  # -> (B, H, W, 3)
    rendered_img = rendered_img.contiguous().view(-1, 3)
    rendered_img[idx] = torch.IntTensor([0, 0, 255]).to(device=device, dtype=rendered_img.dtype)
    rendered_img = rendered_img.view(B, H, W, 3).contiguous()

    rendered_img = rendered_img.permute(0, 3, 1, 2)

    plt.imshow(rendered_img.permute(0, 2, 3, 1)[1].cpu().numpy())

    return rendered_img


def train(cfg):
    # Load model
    model = config.get_model(cfg)

    return model, cfg


def plot_gt_pred(model, data, mode, max_images=5):

    # fw pass to get gt keypoints
    """ Fwd pass through the parametric body model to obtain mesh vertices.

    Args:
           root_loc (torch.Tensor): Root location (B, 10).
        root_orient (torch.Tensor): Root orientation (B, 3).
              betas (torch.Tensor): Shape coefficients (B, 10).
          pose_body (torch.Tensor): Body joint rotations (B, 21*3).
          pose_hand (torch.Tensor): Hand joint rotations (B, 2*3).

    Returns:
        mesh vertices (torch.Tensor): (B, 6890, 3)
    """

    # eval
    model.eval()

    # data to device
    for key in data.keys():
        if torch.is_tensor(data[key]):
            data[key] = data[key].to(device='cuda')

    # predict keypoints
    prediction = model.forward(data)
    pred_vertices = prediction['vertices']

    # get verticies
    gt_vertices = model.get_vertices(data['root_loc'],
                                     data['root_orient'],
                                     data['betas'],
                                     data['pose_body'],
                                     data['pose_hand'])
    print(gt_vertices.shape)

    # project keypoints to meshgrid
    eval_images = None
    ret_images = eval_images is None or eval_images.shape[0] < 10

    if ret_images:
        if mode == 'train':
            data_plot = data['image_crop']
        elif mode == 'val':
            data_plot = data['image']

        gt_images = proj_verts(gt_vertices, data_plot,
                               data['fx'], data['fy'],
                               data['cx'], data['cy'])

        pred_images = proj_verts(pred_vertices,  data_plot,
                                data['fx'], data['fy'],
                                data['cx'], data['cy'])

        images = torch.cat((gt_images, pred_images), dim=2)

    loss_dict = {'v2v_l2': torch.pow(gt_vertices - pred_vertices, 2).mean().item()}
    # eval_step_dict, imgs = compute_val_loss(prediction, model, data, ret_images)

    if eval_images is None:
        eval_images = images[:max_images, ...].cpu()
    elif eval_images.shape[0] < max_images:
        imgs = eval_images[:max_images - eval_images.shape[0], ...].cpu()
        eval_images = torch.cat((eval_images, imgs), dim=0)

    eval_list = defaultdict(list)
    for k, v in loss_dict.items():
        eval_list[k].append(v)

    eval_dict = {k: float(np.mean(v)) for k, v in eval_list.items()}

    # concatenate images along the W dimension
    eval_images = eval_images.permute(2, 0, 3, 1).contiguous()
    eval_images = eval_images.view(eval_images.shape[0], -1, 3).permute(2, 0, 1)

    plt.figure(figsize=(20, 12))
    plt.imshow(eval_images.permute([1, 2, 0]));
    plt.axis('off')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train pipeline.')
    parser.add_argument('--config', type=str, default='../configs/default.yaml',  help='Path to a config file.')
    _args = parser.parse_args()
    print(_args)

    # load model
    model, cfg = train(config.load_config(_args))

    # load ds
    train_data_loader = config.get_data_loader(cfg, mode='train')
    val_data_loader = config.get_data_loader(cfg, mode='val')
    print(f'Len training data: {len(train_data_loader)}')
    print(f'Len val data: {len(val_data_loader)}')

    # take some examples
    data_train = next(iter(train_data_loader))
    data_train.keys()
    data_val = next(iter(val_data_loader))
    data_val.keys()

    # Plot images/gt & pred meshes
    plot_gt_pred(model, data_train, mode='train', max_images=3)
    plot_gt_pred(model, data_val, mode='val', max_images=3)
    print()
