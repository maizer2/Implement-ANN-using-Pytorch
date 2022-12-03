from network.GAN.DCGAN import Generator, Discriminator
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

# --------------------------------------------------------------- #


def train_DCGAN(
    batch_size: int = 2**8,
    img_channels: int = 1,
    num_workers: int = 6,
    num_epochs: int = 5000,
    check_point: int = 20,
    latent_space_vector: int = 100,
    lr: float = 2e-03,
    betas: Tuple[float] = (0.5, 0.999),
    save_root: str = "train/GAN/DCGAN/checkpoint/"

):

    ##################
    # Hyperparameter #
    ##################

    os.makedirs(save_root, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ###########
    # Prepare #
    ###########

    if img_channels == 1:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            transforms.Resize((64, 64))
        ])
    elif img_channels == 3:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((64, 64))
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
        num_workers=num_workers,
        pin_memory=True
    )

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    optim_G = optim.Adam(netG.parameters(), lr, betas)
    optim_D = optim.Adam(netD.parameters(), lr, betas)

    writer = SummaryWriter("Tensorboard/GAN/DCGAN")

    for epoch in tqdm(range(0, num_epochs + 1)):
        for imgs, _ in train_loader:

            x = imgs.to(device)
            fake = netG(torch.randn((x.size(0), latent_space_vector, 1, 1)).to(device))

            y_real = torch.ones((x.size(0), ), dtype=torch.float, device=device)
            y_fake = torch.zeros((x.size(0), ), dtype=torch.float, device=device)

            # Train Discriminator
            
            optim_D.zero_grad()

            y_real_hat = netD(x)
            y_fake_hat = netD(fake)

            loss_real = criterion(y_real_hat, y_real)
            loss_fake = criterion(y_fake_hat, y_fake)

            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optim_D.step()

            # Train Generator

            optim_G.zero_grad()

            fake = netG(torch.randn((x.size(0), latent_space_vector, 1, 1)).to(device))

            y_fake_hat = netD(fake)

            loss_G = criterion(y_fake_hat, y_real)
            loss_G.backward()
            optim_G.step()  

            if epoch % check_point == 0:

                real_grid = make_grid(x[:16], padding=3, normalize=True)
                fake_grid = make_grid(fake[:16], padding=3, normalize=True)

                writer.add_image("DCGAN/Image/Real", real_grid, epoch)
                writer.add_image("DCGAN/Image/Fake", fake_grid, epoch)
                writer.add_scalar("DCGAN/Scalar/Loss/netD", loss_D.item(), epoch)
                writer.add_scalar("DCGAN/Scalar/Loss/netG", loss_G.item(), epoch)

                torch.save(netG.state_dict(), f"{save_root}/{epoch}_netG.pth")
                torch.save(netD.state_dict(), f"{save_root}/{epoch}_netD.pth")
    
    writer.close()