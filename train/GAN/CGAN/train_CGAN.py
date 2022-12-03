from network.GAN.cGAN import Generator, Discriminator
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
    parser.add_argument("--latent_vector", default=100)
    parser.add_argument("--lr", default=1e-7)
    parser.add_argument("--num_workers", default=6)
    parser.add_argument("--num_epochs", default=5000)
    parser.add_argument("--check_point", default=20)
    parser.add_argument("--betas", default=(0.5, 0.999))
    parser.add_argument("--save_root", default="train/CNN/ResNet")
    opt = parser.parse_args()
    return opt

# ---------------------------------------------------------------- #

def train_ResNet():

    opt = get_opt()

    dist.init_process_group("nccl")
    gpu_id = dist.get_rank()

    model_save_root = f"{opt.save_root}/checkpoint/{opt.data_name}/ResNet_{opt.res_layers}"
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

    netG = DDP(Generator().to(gpu_id), [gpu_id], broadcast_buffers=False).apply(weights_init)
    netD = DDP(Discriminator().to(gpu_id), [gpu_id], broadcast_buffers=False).apply(weights_init)

    criterion = nn.CrossEntropyLoss().to(gpu_id)
    optimG = optim.Adam(netG.parameters(), lr=opt.lr, betas=opt.betas)
    optimD = optim.Adam(netD.parameters(), lr=opt.lr, betas=opt.betas)

    writer = SummaryWriter(f"Tensorboard/CNN/ResNet/ResNet{opt.res_layers}")

    for epoch in tqdm(range(0, opt.num_epochs + 1)):
        for imgs, labels in train_loader:
            optimG.zero_grad()

            x = imgs.to(gpu_id)
            z = torch.randn((x.size(0), opt.latent_vector), dtype=torch.float, device=gpu_id)
            condition = 

            y = labels.to(gpu_id)
            y_real = torch.ones((x.size(0), ), dtype=torch.float, device=gpu_id)
            y_fake = torch.zeros((x.size(0), ), dtype=torch.float, device=gpu_id)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            if epoch % opt.check_point == 0:
                writer.add_scalar(f"Loss/ResNet/ResNet{opt.res_layers}/", loss.item(), epoch)
                torch.save(model.state_dict(), f"{model_save_root}/{epoch}_model.pth")
                
    torch.save(model.state_dict(), f"{model_save_root}/final.pth")
    writer.close()
    dist.destroy_process_group()