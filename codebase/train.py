import argparse
from os.path import join
import numpy as np

# import torch
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from checkpoints import CheckpointIO
import config
from time import time
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# print("PyTorch Version: ", torch.__version__)
cudnn.benchmark = True


def train(cfg, model_file):
    # shortened
    out_dir = cfg['out_dir']
    model_file = model_file if model_file is not None else 'model_best.pt'  # continue from last best checkpoint
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    model_selection_metric = cfg['training'].get('model_selection_metric', 'v2v_l2')

    # Load model
    model = config.get_model(cfg)

    # Selet Optimizer
    optimizer = config.get_optimizer(model, cfg)

    # Select Trainer
    trainer = config.get_trainer(model, out_dir, cfg, optimizer)

    # LR secheduler (Reduce LR by factor of 0.1 after every 3 epochs of plateau)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-6,
                                                      verbose=True)
    # Load checkpoint
    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)

    # init datasets
    train_data_loader = config.get_data_loader(cfg, mode='train')
    val_data_loader = config.get_data_loader(cfg, mode='val')

    # Check data
    print(f'Len training data: {len(train_data_loader)}')
    print(f'Len val data: {len(val_data_loader)}')
    # items = next(iter(train_data_loader))
    # items.keys()
    # eval_dict, val_img = trainer.evaluate(items)

    # load pretrained model if any
    load_dict = checkpoint_io.safe_load(model_file)
    epoch_it = load_dict.get('epoch_it', 0)
    it = load_dict.get('it', 0)
    metric_val_best = load_dict.get('loss_val_best', float('inf'))

    # prepare loggers
    print(model, f'\n\nTotal number of parameters: {sum(p.numel() for p in model.parameters()):d}')
    print(f'Current best validation metric: {metric_val_best:.8f}')

    #####################
    # Name the experiment
    #####################
    config.cond_mkdir(out_dir)
    comment = '_[resnet50_dataaug]'  # '_[resnet18]'
    log_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = SummaryWriter(join(out_dir, 'logs', log_time + comment))
    print(f'Running experiment: {logger.file_writer.get_logdir()}')


    # training loop
    t_0 = time()
    bad_epochs = 0
    # while True:
    for epoch in range(1, _args.epochs + 1):
        print('------------------------------------------')
        print(f'Epoch {epoch}/{_args.epochs} - Training running...')
        print('------------------------------------------')

        epoch_it += 1
        t_0_epoch = time()
        for batch in train_data_loader:
            it += 1
            # batch['image_crop'].size(0)

            # Run trainer, get train loss metrics
            loss_dict = trainer.train_step(batch)
            loss = loss_dict['total_loss']

            # Log train loss metric (v2v_l1 / v2v_l2 / total_loss)
            for k, v in loss_dict.items():
                logger.add_scalar(f'train/{k}', v, it)

            # Print output
            if print_every > 0 and (it % print_every) == 0:
                print(f'[Epoch {epoch_it:02d}] it={it:05d}, loss={loss:.8f}')

            # Save checkpoint
            if checkpoint_every != 0 and (it % checkpoint_every) == 0:
                print('Saving checkpoint...')
                checkpoint_io.save(f'model_{it:d}.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)

            # Run validation
            if validate_every > 0 and (it % validate_every) == 0 and it > 0:
                print()
                print('------------------------------------------')
                print(f'Epoch {epoch}/{_args.epochs} - Validation running...')
                print('------------------------------------------')

                bad_epochs = 0
                eval_dict, val_img = trainer.evaluate(val_data_loader)

                # Run val, get eval loss metric
                metric_val = eval_dict[model_selection_metric]
                print(f'Validation metric ({model_selection_metric}): {metric_val:.8f}')

                # Log eval images
                if val_img is not None:
                    logger.add_image(f'val/renderings', val_img, it)

                # Log eval score
                for k, v in eval_dict.items():
                    logger.add_scalar(f'val/{k}', v, it)

                if metric_val < metric_val_best:
                    metric_val_best = metric_val
                    print(f'New best model (loss {metric_val_best:.8f})')
                    checkpoint_io.save(f'{logger.logdir}/model_best.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    print(f'Eval did not improve, since {bad_epochs} epochs.')

                if bad_epochs == _args.early_stop:
                    print(f'Early stopping after {bad_epochs} epochs.')
                    break

        # time to finish one epoch
        time_elapsed = time() - t_0_epoch
        print(f'Epoch completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s.')

    # step learning rate scheduler
    exp_lr_scheduler.step(metric_val)

    time_elapsed = time() - t_0
    print(f'Training completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train pipeline.')
    parser.add_argument('--config', type=str, default='../configs/default.yaml',  help='Path to a config file.')
    parser.add_argument('--model_file', type=str, default=None, help='Overwrite the model path.')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 20)')
    parser.add_argument('--early-stop', type=int, default=5, metavar='N', help='number of iters to stop traing(default: 5)')
    _args = parser.parse_args()
    print(_args)

    train(config.load_config(_args), _args.model_file)
    # smpl_dict = np.load(config.load_config(_args)['data']['bm_path'], encoding='latin1')
    # print(smpl_dict['shapedirs'].shape)
    # # print(smpl_dict['shapedirs'][-1])
    # print(smpl_dict['posedirs'].shape)
    # posedirs = smpl_dict['posedirs']
    # posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
    # print(posedirs.shape)
   # print(smpl_dict[])

    # import h5py
    # mean_values = h5py.File("/home/henry/Downloads/neutral_smpl_mean_params.h5")
    # print('shape values')
    # for i in range(mean_values['shape'].shape[0]):
    #     print(mean_values['shape'][i])
    # print('pose values')
    # for i in range(mean_values['pose'].shape[0]):
    #     print(mean_values['pose'][i])