import argparse
import config
import torch

if __name__ == '__main__':
    ####### TESTING ######
    parser = argparse.ArgumentParser(description='Train pipeline.')
    parser.add_argument('--config', type=str, default='../configs/default.yaml',  help='Path to a config file.')
    _args = parser.parse_args()

    # load model
    cfg = config.load_config(_args)
    model = config.get_model(cfg)

    # load ds
    train_data_loader = config.get_data_loader(cfg, mode='train')
    val_data_loader = config.get_data_loader(cfg, mode='val')

    # take some examples
    data_train = next(iter(train_data_loader))
    data_train.keys()
    data_val = next(iter(val_data_loader))
    data_val.keys()

    model.eval()
    prediction = model.forward(data_train)
    pred_vertices = prediction['vertices']
    print(pred_vertices.shape)

    joints, v_a_pose, f, abs_bone_transforms, bone_transforms, betas = model.get_joints(data_train['root_loc'],
                                        data_train['root_orient'],
                                        data_train['betas'],
                                        data_train['pose_body'],
                                        data_train['pose_hand'])
    print(joints.shape)
    print(v_a_pose.shape)
    print(f.shape)
    print(abs_bone_transforms.shape)
    print(bone_transforms.shape)
    print(betas.shape)

