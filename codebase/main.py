import argparse
import config
import torch
from regressor.model import Discriminator, ConvModel_Pre
from regressor.training import HMRTrainer

if __name__ == '__main__':
    ####### TESTING ######
    parser = argparse.ArgumentParser(description='Train pipeline.')
    parser.add_argument('--config', type=str, default='../configs/default.yaml',  help='Path to a config file.')
    _args = parser.parse_args()

    # load model
    cfg = config.load_config(_args)
    
    out_dir = cfg['out_dir']
    # # load ds
    train_data_loader = config.get_data_loader(cfg, mode='train')
    # val_data_loader = config.get_data_loader(cfg, mode='val')

    # # take some examples
    data_train = next(iter(train_data_loader))
    # data_train.keys()
    # data_val = next(iter(val_data_loader))
    # data_val.keys()

    # model.eval()
    # prediction = model.forward(data_train)
    # pred_vertices = prediction['vertices']
    # print(pred_vertices.shape)


    disc = Discriminator().to(device=cfg['device'])
    gen = config.get_model(cfg, 1)

    e_opt = torch.optim.Adam(
            gen.parameters(),
            lr = 0.0001,
            weight_decay = 0.0001
    )
    d_opt = torch.optim.Adam(
            disc.parameters(),
            lr = 0.001,
            weight_decay = 0.0001
    )
    trainer = HMRTrainer(gen, disc, e_opt, d_opt, out_dir, cfg)
    lol = trainer.train_step(data_train)
    print(lol.items())