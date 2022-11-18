from network.GAN.VanilaGAN import Generator, Discriminator
from network.weights_init import weights_init

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid

import os
from tqdm import tqdm
from torchsummary import summary
from typing import Tuple

# ---------------------------------------------------------------------------------- #

def train_VanilaGAN(
    num_gpus: int = 3,
    use_gpu: int = 0,
    batch_size: int = 128,
    img_channels: int = 1,
    num_workers: int = 4,
    num_epochs: int = 5000,
    check_point : int = 20,
    latent_space_vector: int = 100,
    lr: float = 1e-03,
    betas: Tuple[float] = (0.5, 0.999),
    save_root: str = "train/GAN/VanilaGAN/checkpoint/"
    ):

    device = torch.device(f"cuda:{use_gpu}" if torch.cuda.is_available() and num_gpus > 0 else "cpu")

    if img_channels == 1:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
    elif img_channels == 3:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    train_data = datasets.MNIST(
        root="/data/DataSet/",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss().to(device)
    optim_G = optim.Adam(netG.parameters(), lr=lr, betas=betas)
    optim_D = optim.Adam(netD.parameters(), lr=lr, betas=betas)

    writer = SummaryWriter("Tensorboard/VanilaGAN")

    for epoch in tqdm(range(0, num_epochs + 1)):
        for imgs, _ in train_loader:

            x = imgs.to(device)
            fake = netG(torch.randn((batch_size, latent_space_vector)))

            y_real = torch.ones((batch_size, ), dtype= torch.cuda.float, device=device)
            y_fake = torch.zeros((batch_size, ), dtype=torch.cuda.float, device=device)

            # Train Discriminator

            optim_D.zero_grad()

            y_real_hat = netD(x)
            y_fake_hat = netG(fake)

            loss_real = criterion(y_real_hat, y_real)
            loss_fake = criterion(y_fake_hat, y_fake)

            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optim_D.step()

            # Train Generator

            optim_G.zero_grad()

            fake = netG(torch.randn((batch_size, latent_space_vector)))

            y_fake_hat = netD(fake)

            loss_G = criterion(y_fake_hat, y_real)
            loss_G.backward()
            optim_G.step()

            if epoch % check_point == 0:

                real_grid = make_grid(x[:16], padding=3, normalize=True)
                fake_grid = make_grid(fake[:16], padding=3, normalize=True)

                writer.add_image("VanilaGAN/Image/Real", real_grid, epoch)
                writer.add_image("VanilaGAN/Image/Fake", fake_grid, epoch)
                writer.add_scalar("vanilaGAN/Scalar/Loss/netG", loss_G.item(), epoch)
                writer.add_scalar("vanilaGAN/Scalar/Loss/netD", loss_D.item(), epoch)

    writer.close()
    os.makedirs(save_root, exist_ok=True)
    torch.save(netG.state_dict(), f"{save_root}/netG.pth")
    torch.save(netD.state_dict(), f"{save_root}/netD.pth")


            
