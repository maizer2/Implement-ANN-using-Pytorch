from network.VAE.vanila_VAE import VAE
from network.weights_init import weights_init

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision
import torchvision.utils as utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
from typing import Tuple, NamedTuple
from collections import namedtuple

# -------------------------------------------------------------------------------------------- #

def train_VAE(
    num_gpus: int = 3,
    use_gpu: int = 0,
    batch_size: int = 512,
    img_channels: int = 3,
    num_workers: int = 4,
    num_epochs: int = 1000,
    check_point: int = 20,
    lr: float = 1e-2,
    betas: Tuple[float] = (0.5, 0.999),
    save_root: str = "train/GAN/VAE/checkpoint/"
):
    os.makedirs(f"{save_root}/", exist_ok=True)
    device = torch.device(f"cuda:{use_gpu}" if torch.cuda.is_available() and num_gpus > 0 else "cpu")

    if img_channels == 1:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            transforms.Resize((224, 224))
        ])
    elif img_channels == 3:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((224, 224))
        ])

    train_data = datasets.MNIST(
        root="/data/DataSet/",
        train=True,
        transform=transform,
        download=True
    )

    train_loader = data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    model = VAE(img_channels).to(device)
    model.apply(weights_init)
    model.train()

    # summary(model, input_size=(img_channels, 224, 224), device="cpu")

    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter("Tensorboard/VAE/")

    for epoch in tqdm(range(0, num_epochs + 1)):
        for imgs, _ in train_loader:
            optimizer.zero_grad()

            x = imgs.to(device)
            noise = torch.randn_like(x, dtype=torch.float, device=device)

            fake = model(noise)
            loss = criterion(x, fake)
            loss.backward()
            optimizer.step()

            if epoch % check_point == 0:
                real_grid = utils.make_grid(x[:16], padding=3, normalize=True)
                fake_grid = utils.make_grid(fake[:16], padding=3, normalize=True)

                writer.add_image(f"VAE/Image/Real/batch:{batch_size},lr:{lr}", real_grid, epoch)
                writer.add_image(f"VAE/Image/Fake/batch:{batch_size},lr:{lr}", fake_grid, epoch)
                writer.add_scalar(f"VAE/Loss/batch:{batch_size}, lr:{lr}", loss.item(), epoch)

    writer.close()
    os.makedirs(save_root, exist_ok=True)
    torch.save(model.state_dict(), f"{save_root}/{epoch}_model.pth")
