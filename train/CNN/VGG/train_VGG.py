from network.CNN.VGG import VGG
from network.weights_init import weights_init

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist

import torchvision
import torchvision.utils as utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import argparse
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from typing import Tuple, NamedTuple
from collections import namedtuple

# ---------------------------------------------------------------- #

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="MNIST")
    parser.add_argument("--data_root", default="/data/DataSet")
    parser.add_argument("--img_channels", default=1)
    parser.add_argument("--img_size", default=(224, 224))
    parser.add_argument("--img_mean", default=(0.5, ))
    parser.add_argument("--img_std", default=(0.5, ))
    parser.add_argument("--batch_size", default=2**5)
    parser.add_argument("--lr", default=1e-7)
    parser.add_argument("--num_workers", default=6)
    parser.add_argument("--num_epochs", default=5000)
    parser.add_argument("--check_point", default=20)
    parser.add_argument("--betas", default=(0.5, 0.999))
    parser.add_argument("--vgg_layers", default=152) #34, 50, 101, 152
    parser.add_argument("--save_root", default="train/CNN/VGG")
    opt = parser.parse_args()
    return opt

# ------------------------------------------------------------------------ #

def get_vgg_config(vgg_layers):

    if vgg_layers == vgg_layers:
        vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    elif vgg_layers == vgg_layers:
        vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    
    elif vgg_layers == vgg_layers:
        vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

    else: # layers == 11 or enter wrong number
        vgg_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    return vgg_config

# ---------------------------------------------------------------- #

def train_VGG():

    opt = get_opt()

    dist.init_process_group("nccl")
    gpu_id = dist.get_rank()

    model_save_root = f"{opt.save_root}/checkpoint/{opt.data_name}/VGG_{opt.vgg_layers}"
    os.makedirs(model_save_root, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(opt.img_mean, opt.img_std),
        transforms.Resize(opt.img_size)
    ])

    train_data = datasets.MNIST(
        root=opt.data_root,
        train=True,
        transform=transform,
        download=True
    )

    train_loader = data.DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=data.DistributedSampler(train_data),
        num_workers=opt.num_workers,
        pin_memory=True
    )

    labels_temp = train_data.class_to_idx
    labels_map = dict(zip(labels_temp.values(), labels_temp.keys()))
        
    vgg_config = get_vgg_config(opt.vgg_layers)
    model = VGG(vgg_config).to(gpu_id).apply(weights_init)

    criterion = nn.CrossEntropyLoss().to(gpu_id)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=opt.betas)

    writer = SummaryWriter(f"Tensorboard/CNN/VGG/VGG{opt.vgg_layers}")

    for epoch in tqdm(range(0, opt.num_epochs + 1)):
        for imgs, labels in train_loader:
            optimizer.zero_grad()

            x = imgs.to(gpu_id)
            y = labels.to(gpu_id)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            if epoch % opt.check_point == 0:
                writer.add_scalar(f"Loss/VGG/VGG{opt.vgg_layers}", loss.item(), epoch)
                torch.save(model.state_dict(), f"{model_save_root}/{epoch}_model.pth")
                
    torch.save(model.state_dict(), f"{model_save_root}/final.pth")
    writer.close()
    dist.destroy_process_group()