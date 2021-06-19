import os
import yaml
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from data.dataset import H36MDataset
from regressor.model import BaseModel
from regressor.training import ConvTrainer

from torch import optim


def load_config(args):
    """ Loads configuration file.

    Returns:
        cfg (dict): configuration file
    """
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def cond_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_data_loader(cfg, mode='train'):
    assert mode in ['train', 'val', 'test']

    if mode == 'train':
        subjects = cfg['data']['train_subjects'].split(',')
        batch_size = cfg['training']['batch_size']
    elif mode == 'val':
        subjects = cfg['data']['val_subjects'].split(',')
        batch_size = cfg['training']['batch_size']
    else:
        subjects = ['S9', 'S11']  # ['S9', 'S11']
        batch_size = 1

    dataset = H36MDataset(dataset_folder=cfg['data']['dataset_folder'],
                          img_folder=cfg['data']['img_folder'],
                          subjects=subjects,
                          mode=mode,
                          img_size=(512, 512))


    if cfg['data'].get('downsample', False):
        denominator = cfg['data'].get('ds_ratio', 0.2)
        indices = range(0, len(dataset), int(1 / denominator))
        subset = Subset(dataset, indices=indices)

    # remove mode=='train' to subsample the val set too
    data_loader = DataLoader(subset if cfg['data'].get('downsample', False) and mode in ['train','val'] else dataset,
                             batch_size=batch_size,
                             num_workers=cfg['training'].get('num_workers', 0),
                             shuffle=mode == 'train',
                             collate_fn=dataset.collate_fn)

    # data_loader = DataLoader(dataset,
    #                          batch_size=batch_size,
    #                          num_workers=cfg['training'].get('num_workers', 0),
    #                          shuffle=mode == 'train',
    #                          collate_fn=dataset.collate_fn,
    #                          drop_last=False)

    return data_loader

# Load the model
def get_model(cfg):
    model = BaseModel.create_model(cfg)
    return model.to(device=cfg['device'])


def get_optimizer(model, cfg):
    """ Create an optimizer. """

    if cfg['training']['optimizer']['name'] == 'SGD':
        # optimizer = optim.SGD(model.parameters(), lr=cfg['training']['optimizer'].get('lr', 1e-4))
        optimizer = optim.SGD(model.parameters(), lr=cfg['training']['optimizer'].get('lr', 1e-4), momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif cfg['training']['optimizer']['name'] == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=cfg['training']['optimizer'].get('lr', 1e-4))
    else:
        raise Exception('Not supported.')

    return optimizer


def get_trainer(model, vis_dir, cfg, optimizer=None):
    """ Create a trainer instance. """

    if cfg['trainer'] == 'conv':
        trainer = ConvTrainer(model, optimizer, vis_dir, cfg)
    else:
        raise Exception('Not supported.')

    return trainer
