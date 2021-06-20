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


def train(cfg, gen_file, disc_file):
    # shortened
    out_dir = cfg['out_dir']
    gen_file = gen_file if gen_file is not None else 'generator_best.pt'  # continue from last best checkpoint
    disc_file = disc_file if disc_file is not None else 'discriminator_best.pt'  # continue from last best checkpoint
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    model_selection_metric = cfg['training'].get('model_selection_metric', 'v2v_l2')

    # Load model
    generator = config.get_model(cfg, 1)
    discriminator = config.get_model(cfg, 0)

    # Selet Optimizer
    gen_opt, disc_opt = config.get_optimizer_gan(generator, discriminator, cfg)

    # Select Trainer
    trainer = config.get_trainer_gan(generator, discriminator, out_dir, cfg, gen_opt, disc_opt)

    # LR secheduler (Reduce LR by factor of 0.1 after every 3 epochs of plateau)
    gen_exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(gen_opt, mode='min', factor=0.1, patience=2, min_lr=1e-6,
                                                      verbose=True)
    disc_exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(disc_opt, mode='min', factor=0.1, patience=2, min_lr=1e-6,
                                                      verbose=True)
    # Load checkpoint
    gen_checkpoint = CheckpointIO(out_dir, model=generator, optimizer=gen_opt)
    disc_checkpoint = CheckpointIO(out_dir, model=discriminator, optimizer=disc_opt)
    # init datasets
    train_data_loader = config.get_data_loader(cfg, mode='train')
    # val_data_loader = config.get_data_loader(cfg, mode='val')

    # Check data
    print(f'Len training data: {len(train_data_loader)}')
    # print(f'Len val data: {len(val_data_loader)}')
    # items = next(iter(train_data_loader))
    # items.keys()
    # eval_dict, val_img = trainer.evaluate(items)

    # load pretrained model if any
    load_dict = gen_checkpoint.safe_load(gen_file)
    
    load_dict = disc_checkpoint.safe_load(disc_file)
    epoch_it = load_dict.get('epoch_it', 0)
    it = load_dict.get('it', 0)
    loss_best = load_dict.get('loss_best', float('inf'))

    # prepare loggers
    # print(generator, f'\n\nTotal number of parameters: {sum(p.numel() for p in generator.parameters()):d}')
    # print("------------------------------------------------------------")
    # print(discriminator, f'\n\nTotal number of parameters: {sum(p.numel() for p in discriminator.parameters()):d}')
    print(f'\n\nTotal number of parameters: {sum(p.numel() for p in generator.parameters()):d}')
    print("---------------------------------------------------------------------")
    print(f'\n\nTotal number of parameters: {sum(p.numel() for p in discriminator.parameters()):d}')
    print(f'Current best loss (v2v_l2): {loss_best:.8f}')

    #####################
    # Name the experiment
    #####################
    config.cond_mkdir(out_dir)
    comment = '_[resnet50_batch20_GAN_fulltraining_2djoint_0.5downsample]'  # '_[resnet18]'
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
            loss = loss_dict['v2v_l2']

            # Log train loss metric (v2v_l1 / v2v_l2 / total_loss)
            for k, v in loss_dict.items():
                logger.add_scalar(f'train/{k}', v, it)

            # Print output
            if print_every > 0 and (it % print_every) == 0:
                print(f'[Epoch {epoch_it:02d}] it={it:05d}, loss={loss:.8f}')

            # Save checkpoint
            if checkpoint_every != 0 and (it % checkpoint_every) == 0:
                # print('Checkpoint...')

                if loss < loss_best:
                    loss_best = loss
                    print(f'New best generator(loss {loss_best:.8f})')
                    gen_checkpoint.save(f'{out_dir}/{it:05d}_generator_best.pt', epoch_it=epoch_it, it=it, loss_best=loss_best)
                    disc_checkpoint.save(f'{out_dir}/{it:05d}_discriminator_best.pt', epoch_it=epoch_it, it=it, loss_best=loss_best)


        # time to finish one epoch
        time_elapsed = time() - t_0_epoch
        print(f'Epoch completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s.')

    # step learning rate scheduler
    gen_exp_lr_scheduler.step(loss)
    disc_exp_lr_scheduler.step(loss)

    time_elapsed = time() - t_0
    print(f'Training completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train pipeline.')
    parser.add_argument('--config', type=str, default='../configs/default.yaml',  help='Path to a config file.')
    parser.add_argument('--gen_file', type=str, default=None, help='Overwrite the generator path.')
    parser.add_argument('--disc_file', type=str, default=None, help='Overwrite the discriminator path')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 20)')
    parser.add_argument('--early-stop', type=int, default=5, metavar='N', help='number of iters to stop traing(default: 5)')
    _args = parser.parse_args()
    print(_args)

    train(config.load_config(_args), _args.gen_file, _args.disc_file)
    # cfg = config.load_config(_args)
    # model = config.get_model(cfg, 1)
    # import os
    # cwd = os.getcwd()
    # print(cwd)
