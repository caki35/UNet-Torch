import torch
from DataLoader import Data_Reg_Binary, Data_Binary, Data_Reg_Fourier1, Data_Reg_Fourier1_2
import loss
from torchvision.utils import make_grid, save_image
import random
from Model import UNet, UNet_multitask, UNet_attention, UNet_fourier1, UNet_fourier1_2, UNet_BS
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
from TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from test import test_single, get_image_list
import pandas as pd

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='the config path')
    args = ap.parse_args()
    return args


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        save_image(ulti, 'train_batch.png')

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
    input_size = (cfg['model_config']['input_size'][0],
                  cfg['model_config']['input_size'][1])  # h, w
    num_class = cfg['model_config']['num_class']
    ch = cfg['model_config']['channel']
    initial_filter_size = cfg['model_config']['initial_filter_size'][0]
    kernel_size = cfg['model_config']['kernel'][0]
    model_type = cfg['model_config']['model_type']
    dropout = cfg['model_config']['dropout']
    dropout_p = cfg['model_config']['drop_out_rate'][0]

    # train configs
    batch_size = cfg['train_config']['batch_size'][0]
    num_workers = cfg['train_config']['num_workers']
    lr_rate = cfg['train_config']['lr_rate'][0]
    Epoch = cfg['train_config']['epochs']
    use_cuda = cfg['train_config']['use_cuda']
    loss_function = cfg['train_config']['loss']
    accuracy_metric = cfg['train_config']['accuracy']
    weight_decay = cfg['train_config']['weight_decay'][0]
    loss.CLASS_NUMBER = cfg['model_config']['num_class']
    # dataset configs
    train_path = cfg['dataset_config']['train_path']
    val_path = cfg['dataset_config']['val_path']
    test_path = cfg['dataset_config']['test_path']
    if test_path:
        test_image_list = get_image_list(test_path[0])
        resultsDict = {}
    else:
        test_image_list = False
    save_dir = cfg['dataset_config']['save_dir']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)

    anydepth = cfg['model_config']['anydepth']
    seeds = cfg['train_config']['seed']
    for currentSeed in seeds:
        output_save_dir = save_dir+ '_seed'+str(currentSeed)
        output_save_dir = os.path.join(save_dir,output_save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        seed_everything(currentSeed)
        if model_type == 'single':
            train_dataset = Data_Binary(
                train_path, ch, anydepth, cfg['dataset_config']['augmentation'], input_size=input_size)
            val_dataset = Data_Binary(
                val_path, ch, anydepth, cfg['dataset_config']['augmentation'], input_size=input_size)

            model = UNet(ch, num_class, initial_filter_size,
                        use_cuda, dropout, dropout_p)
            # model = UNet_BS([1, 32, 64, 128, 256, 512],
            #                  "parameters", "dropout")
            # model.apply(weights_init)
        elif model_type == 'TransUnet':
            
            train_dataset = Data_Binary(
                train_path, ch, anydepth, cfg['dataset_config']['augmentation'], input_size=input_size)
            val_dataset = Data_Binary(
                val_path, ch, anydepth, cfg['dataset_config']['augmentation'], input_size=input_size)
            
            config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
            config_vit.n_classes = cfg['model_config']['num_class']
            config_vit.n_skip = 3
            config_vit.patches.grid = (int(input_size[1] / 16), int(input_size[1] / 16))

            model = ViT_seg(config_vit, img_size=input_size[0], num_classes=cfg['model_config']['num_class']).cuda()
            model.load_from(weights=np.load("TransUnet/R50+ViT-B_16.npz"))
        elif model_type == 'multi_task':
            train_dataset = Data_Reg_Binary(
                train_path, ch, anydepth, input_size=input_size)
            val_dataset = Data_Reg_Binary(
                val_path, ch, anydepth, input_size=input_size)
            model = UNet_multitask(ch, num_class, initial_filter_size, use_cuda)

        elif model_type == 'attention':
            train_dataset = Data_Binary(
                train_path, ch, anydepth, input_size=input_size)
            val_dataset = Data_Binary(
                val_path, ch, anydepth, input_size=input_size)
            model = UNet_attention(ch, num_class, initial_filter_size,
                        use_cuda, dropout, dropout_p)


        elif model_type == 'fourier1':
            train_dataset = Data_Reg_Fourier1(
                train_path, ch, anydepth, input_size=input_size)
            val_dataset = Data_Reg_Fourier1(
                val_path, ch, anydepth, input_size=input_size)
            model = UNet_fourier1(ch, num_class, initial_filter_size, use_cuda)
        elif model_type == 'fourier1_2':
            train_dataset = Data_Reg_Fourier1_2(
                train_path, ch, anydepth, input_size=input_size)
            val_dataset = Data_Reg_Fourier1_2(
                val_path, ch, anydepth, input_size=input_size)
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
        # check_input(dataloaders)
        # # optimizers
        if cfg['train_config']['optimizer'] == 'Adam':
            optimizer = optim.Adam(
                model.parameters(), lr=lr_rate, weight_decay=weight_decay)
        elif cfg['train_config']['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError('Invalid otpimizer "%s"' % cfg['train_config']['optimizer'])

        if accuracy_metric in ['dice_score']:
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=30, min_lr=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-5)
        trainer = Trainer(model, model_type, dtype, device, output_save_dir, dataloaders, batch_size, optimizer,
                        patience=30, num_epochs=Epoch, loss_function=loss_function, accuracy_metric=accuracy_metric, lr_scheduler=False, start_epoch=start_epoch)
        best_model = trainer.train()
        if test_image_list:
            print('Testing best model:')
            currResultsDict = test_single(trainer.model, device, input_size, ch, cfg['model_config']['num_class'], test_image_list, output_save_dir)
            resultsDict[currentSeed] = currResultsDict
    
    df = pd.DataFrame(resultsDict)
    df = df.transpose()
    df.to_csv(os.path.join(save_dir,'results.csv'))

if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    # config_path = 'config.yml'
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    main(cfg)
