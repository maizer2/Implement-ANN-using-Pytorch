from network.GAN.dcgan import Generator, Discriminator, weights_init

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple

# --------------------------------------------------------------- #


def train_dcgan(
    num_gpus: int = 3,
    use_gpu: int = 0,
    batch_size: int = 15000,
    img_channels: int = 3,
    num_workers: int = 4,
    num_epochs: int = 5000,
    check_point: int = 200,
    latent_space_vector: int = 100,
    lr: float = 0.0002,
    betas: Tuple[float] = (0.5, 0.999),
    save_root: str = "train/GAN/DCGAN/checkpoint/"

):
    ##################
    # Hyperparameter #
    ##################

    device = torch.device(f"cuda:{use_gpu}" if torch.cuda.is_available() and num_gpus > 0 else "cpu")

    ###########
    # Prepare #
    ###########

    if img_channels == 3:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((227, 227))
        ])
    elif img_channels == 1:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            transforms.Resize((227, 227))
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
        num_workers=num_workers
    )

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # if (device.type == "cuda") and (num_gpus > 1):
    #     netG = nn.DataParallel(netG, list(range(num_gpus)))
    #     netD = nn.DataParallel(netD, list(range(num_gpus)))

    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    gen_opt = optim.Adam(
        params=netG.parameters(),
        lr=lr,
        betas=betas
    ).to(device)
    disc_opt = optim.Adam(
        params=netD.parameters(),
        lr=lr,
        betas=betas
    ).to(device)

    #########
    # Train #
    #########

    writer = SummaryWriter("Tensorboard/DCGAN")

    for epoch in tqdm(range(0, num_epochs + 1)):
        for idx, (imgs, _) in enumerate(train_loader):
            disc_opt.zero_grad(), gen_opt.zero_grad()

            ###########
            # Prepard #
            ###########

            x = imgs.to(device)
            y_real = torch.ones((x.size(0), ), dtype=torch.float, device=device)
            y_fake = torch.zeros((x.size(0), ), dtype=torch.float, device=device)


            ##########################
            # Training Discriminator #
            ##########################

            noise = torch.randn((x.size(0), latent_space_vector, 1, 1), dtype=torch.float, device=device)
            fake = netG(noise)
            
            ####################################
            # 1. Discriminate real imgs        #
            # 2. Criterion real and real label #
            ####################################

            y_hat = netD(x).view(-1)
            loss_real = criterion(y_hat, y_real)

            ####################################
            # 1. Discriminate fake imgs        #
            # 2. Criterion fake and fake label #
            ####################################

            y_hat = netD(fake).view(-1)
            loss_fake = criterion(y_hat, y_fake)

            ########################################################
            # 1. Calculate loss avarage of loss_fake and loss_real #
            # 2. Update gradients                                  #
            # 3. Update optimizer                                  #
            ########################################################

            disc_loss = (loss_real + loss_fake) / 2

            disc_loss.backward(retain_graph=True)
            disc_opt.step()
            
            ######################
            # Training Generator #
            ######################

            noise = torch.randn((x.size(0), latent_space_vector, 1, 1), dtype=torch.float, device=device)
            fake = netG(noise)

            ####################################
            # 1. Discriminate fake imgs        #
            # 2. Criterion fake and real label #
            ####################################

            y_hat = netD(fake).view(-1)
            gen_loss = criterion(y_hat, y_real)

            ##############################
            # 1. Update gradients        #
            # 2. Update optimizer        #
            ##############################

            gen_loss.backward()
            gen_opt.step()  

            ################################
            # Visualization for x and fake #
            # 1. make_grid(x, fake)        #
            # 2. tensorboard add_images()  #
            # 3. tensorboard add_scalar()  #
            ################################

            if epoch % check_point == 0:

                real_grid = torchvision.utils.make_grid(x[:16], padding=3, normalize=True)
                fake_grid = torchvision.utils.make_grid(fake[:16], padding=3, normalize=True)

                writer.add_image("Real", real_grid, epoch)
                writer.add_image("Fake", fake_grid, epoch)
                writer.add_scalar("Loss/Discriminator", disc_loss.item(), epoch)
                writer.add_scalar("Loss/Generator", gen_loss.item(), epoch)
    
    writer.close()
    os.makedirs(save_root, exist_ok=True)
    torch.save(netG.state_dict(), f"{save_root}/netG.pth")
    torch.save(netD.state_dict(), f"{save_root}/netD.pth")