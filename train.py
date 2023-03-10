import torch
from DataLoader import Data_Reg_Binary, Data_Binary, Data_Reg_Fourier1, Data_Reg_Fourier1_2
from loss import calc_loss, MultitaskUncertaintyLoss
from torchvision.utils import make_grid, save_image

from Model import UNet, UNet_multitask, UNet_attention, UNet_fourier1, UNet_fourier1_2
from collections import defaultdict
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import os
import time
import numpy as np
import argparse
import yaml
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Trainer import Trainer

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='the config path')
    args = ap.parse_args()
    return args


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    return ("{}: {}".format(phase, ", ".join(outputs)))


def plot_loss_functions(output_save_dir, train_loss, val_loss, name):
    plt.figure(figsize=(8, 4))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(np.arange(len(train_loss)), train_loss, label='train loss')
    plt.plot(np.arange(len(val_loss)), val_loss, label='val loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_save_dir, '{}.png'.format(name)))
    plt.cla()


def check_input(dataloaders, titles=["Input", 'Target']):
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    if len(train_batch) == 3:
        img, target, dist = train_batch
        number_of_batch = img.shape[0]
        print('image shape:', img.shape)
        print('binary label shape:', target.shape)
        print('dist label shape:', dist.shape)
        grid_img1 = make_grid(img)
        grid_img2 = make_grid(target)
        grid_img3 = make_grid(dist)
        ulti = make_grid([grid_img1, grid_img2, grid_img3], nrow=1)
        save_image(ulti,'train_batch.png')

        img, target, dist = val_batch
        number_of_batch = img.shape[0]
        print('image shape:', img.shape)
        print('binary label shape:', target.shape)
        print('dist label shape:', dist.shape)
        grid_img1 = make_grid(img)
        grid_img2 = make_grid(target)
        grid_img3 = make_grid(dist)
        ulti = make_grid([grid_img1, grid_img2, grid_img3], nrow=1)
        save_image(ulti, 'val_batch.png')

    elif len(train_batch) == 4:
        img, target, dist1, dist2 = train_batch
        number_of_batch = img.shape[0]
        print('image shape:', img.shape)
        print('binary label shape:', target.shape)
        print('dist1 label shape:', dist1.shape)
        print('dist2 label shape:', dist2.shape)

        grid_img1 = make_grid(img)
        grid_img2 = make_grid(target)
        grid_img3 = make_grid(dist1)
        grid_img4 = make_grid(dist2)
        ulti = make_grid([grid_img1, grid_img2, grid_img3, grid_img4], nrow=1)
        save_image(ulti, 'train_batch.png')

        img, target, dist1, dist2 = val_batch
        number_of_batch = img.shape[0]
        print('image shape:', img.shape)
        print('binary label shape:', target.shape)
        print('dist1 label shape:', dist1.shape)
        print('dist2 label shape:', dist2.shape)

        grid_img1 = make_grid(img)
        grid_img2 = make_grid(target)
        grid_img3 = make_grid(dist1)
        grid_img4 = make_grid(dist2)

        ulti = make_grid([grid_img1, grid_img2, grid_img3, grid_img4], nrow=1)
        save_image(ulti, 'val_batch.png')


    else:
        img, target = train_batch
        number_of_batch = img.shape[0]
        print('image shape:', img.shape)
        print('binary label shape:', target.shape)
        grid_img1 = make_grid(img)
        grid_img2 = make_grid(target)
        ulti = make_grid([grid_img1, grid_img2], nrow=1)
        save_image(ulti, 'train_batch.png')

        img, target = val_batch
        number_of_batch = img.shape[0]
        print('image shape:', img.shape)
        print('binary label shape:', target.shape)
        grid_img1 = make_grid(img)
        grid_img2 = make_grid(target)
        ulti = make_grid([grid_img1, grid_img2], nrow=1)
        save_image(ulti, 'val_batch.png')


def main(cfg):

    # model configs
    input_size = (cfg['model_config']['input_size'][1],
                  cfg['model_config']['input_size'][0])
    num_class = cfg['model_config']['num_class']
    ch = cfg['model_config']['channel']
    initial_filter_size = cfg['model_config']['initial_filter_size'][0]
    kernel_size = cfg['model_config']['kernel'][0]
    model_type = cfg['model_config']['model_type']
    dropout = cfg['model_config']['dropout']
    dropout_p = 0.25

    # train configs
    batch_size = cfg['train_config']['batch_size'][0]
    num_workers = cfg['train_config']['num_workers']
    lr_rate = cfg['train_config']['lr_rate'][0]
    Epoch = cfg['train_config']['epochs']
    use_cuda = cfg['train_config']['use_cuda']
    loss_function = cfg['train_config']['loss']
    accuracy_metric = cfg['train_config']['accuracy']
    weight_decay = cfg['train_config']['weight_decay'][0]

    # dataset configs
    train_path = cfg['dataset_config']['train_path']
    val_path = cfg['dataset_config']['val_path']
    aug_rate = cfg['dataset_config']['aug_rate']
    output_save_dir = cfg['dataset_config']['save_dir']

    
    if model_type == 'single':
        train_dataset = Data_Binary(
            train_path, ch, input_size=input_size)
        val_dataset = Data_Binary(val_path, ch, input_size=input_size)

        model = UNet(ch, num_class, initial_filter_size,
                     use_cuda, dropout, dropout_p)
        
    elif model_type == 'multi_task':
        train_dataset = Data_Reg_Binary(
                    train_path, ch, input_size=input_size)
        val_dataset = Data_Reg_Binary(val_path, ch, input_size=input_size)
        model = UNet_multitask(ch, num_class, initial_filter_size, use_cuda)

    elif model_type == 'attention':
        train_dataset = Data_Binary(
            train_path, ch, input_size=input_size)
        val_dataset = Data_Binary(val_path, ch, input_size=input_size)
        model = UNet_attention(ch, num_class, initial_filter_size, use_cuda)

    elif model_type == 'fourier1':
        train_dataset = Data_Reg_Fourier1(
            train_path, ch, input_size=input_size)
        val_dataset = Data_Reg_Fourier1(val_path, ch, input_size=input_size)
        model = UNet_fourier1(ch, num_class, initial_filter_size, use_cuda)
    elif model_type == 'fourier1_2':
        train_dataset = Data_Reg_Fourier1_2(
            train_path, ch, input_size=input_size)
        val_dataset = Data_Reg_Fourier1_2(val_path, ch, input_size=input_size)
        model = UNet_fourier1_2(ch, num_class, initial_filter_size, use_cuda)

    else:
        raise ValueError('Invalid model_type "%s"' % model_type)
    
    start_epoch = 1
    if cfg['resume']['flag']:
        model.load_state_dict(torch.load(cfg['resume']['path']))
        start_epoch = cfg['resume']['epoch']
    if use_cuda:
        print('Gpu available')
        print(torch.cuda.get_device_name(0))
        device = "cuda:0"
        dtype = torch.cuda.FloatTensor
        model.to(device=device)
    else:
        model.to(device="cpu")

    print(model)
    print('Train set size:', len(train_dataset))
    print('Val set size:', len(val_dataset))
    print('Loss Function:', loss_function)
    train_loader = DataLoader(
        train_dataset, batch_size,
        shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    check_input(dataloaders)
    # optimizers
    optimizer = optim.Adam(
        model.parameters(), lr=lr_rate, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=30, min_lr = 1e-5)

    trainer = Trainer(model, model_type, dtype, device, output_save_dir, dataloaders, batch_size, optimizer,
                      patience=30, num_epochs=Epoch, loss_function=loss_function, accuracy_metric=accuracy_metric, lr_scheduler=lr_scheduler, start_epoch=start_epoch)
    best_model = trainer.train()
    

if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    # config_path = 'config.yml'
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    main(cfg)
