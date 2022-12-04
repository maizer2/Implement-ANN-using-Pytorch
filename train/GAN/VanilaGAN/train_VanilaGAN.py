from network.GAN.VanilaGAN import Generator, Discriminator
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
    parser.add_argument("--img_size", default=(28, 28), help="image size in tuple type")
    parser.add_argument("--img_mean", default=(0.5, ))
    parser.add_argument("--img_std", default=(0.5, ))
    parser.add_argument("--batch_size", default=2**5)
    parser.add_argument("--latent_vector", default=100)
    parser.add_argument("--lr", default=1e-7)
    parser.add_argument("--num_workers", default=6)
    parser.add_argument("--num_epochs", default=400)
    parser.add_argument("--check_point", default=20)
    parser.add_argument("--betas", default=(0.5, 0.999))
    parser.add_argument("--save_root", default="train/GAN/VanilaGAN")
    opt = parser.parse_args()
    return opt

# --------------------------------------------------------------- #

def train_VanilaGAN():

    opt = get_opt()

    dist.init_process_group("nccl")
    gpu_id = dist.get_rank()

    model_save_root = f"{opt.save_root}/checkpoint/{opt.data_name}"
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

    criterion = nn.BCELoss().to(gpu_id)
    optimG = optim.Adam(netG.parameters(), lr=opt.lr, betas=opt.betas)
    optimD = optim.Adam(netD.parameters(), lr=opt.lr, betas=opt.betas)

    writer = SummaryWriter(f"Tensorboard/GAN/VanilaGAN")

    for epoch in tqdm(range(0, opt.num_epochs + 1)):
        for imgs, _ in train_loader:

            x = imgs.to(gpu_id)
            y_real = torch.ones((x.size(0), ), dtype=torch.float, device=gpu_id)
            y_fake = torch.zeros((x.size(0), ), dtype=torch.float, device=gpu_id)

            # Training Discriminator
            z = torch.randn((x.size(0), opt.latent_vector), dtype=torch.float, device=gpu_id)

            optimD.zero_grad()

            x_hat = netG(z)

            y_real_hat = netD(x)
            y_fake_hat = netD(x_hat)

            loss_real = criterion(y_real_hat, y_real)
            loss_fake = criterion(y_fake_hat, y_fake)

            lossD = (loss_real + loss_fake) / 2
            lossD.backward()
            optimD.step()

            # Training Generator
            z = torch.randn((x.size(0), opt.latent_vector), dtype=torch.float, device=gpu_id)

            optimG.zero_grad()

            x_hat = netG(z)

            y_fake_hat = netD(x_hat)

            lossG = criterion(y_fake_hat, y_real)
            lossG.backward()
            optimG.step()

            if epoch % opt.check_point == 0:
                
                real_grid = utils.make_grid(x[:16], padding=3, normalize=True)
                fake_grid = utils.make_grid(x_hat[:16], padding=3, normalize=True)

                writer.add_image("CDGAN/Image/Real", real_grid, epoch)
                writer.add_image("VanilaGAN/Image/Fake", fake_grid, epoch)

                writer.add_scalar(f"DCGAN/Scalar/Loss/netG", lossG.item(), epoch)
                writer.add_scalar(f"DCGAN/Scalar/Loss/netD", lossD.item(), epoch)
                torch.save(netG.state_dict(), f"{model_save_root}/{epoch}_netG.pth")
                torch.save(netD.state_dict(), f"{model_save_root}/{epoch}_netD.pth")
                
    torch.save(netG.state_dict(), f"{model_save_root}/final.pth")
    torch.save(netD.state_dict(), f"{model_save_root}/final.pth")
    writer.close()
    dist.destroy_process_group()


            
