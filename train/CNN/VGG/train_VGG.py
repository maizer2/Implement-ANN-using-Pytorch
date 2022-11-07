from network.CNN.VGG import VGG
from network.weights_init import weights_init

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple

# ------------------------------------------------------------------------ #

def train_VGG(
    num_gpus: int = 3,
    use_gpu: int = 0,
    batch_size: int = 12,
    img_channels: int = 3,
    vgg_layers: int = 11,
    num_workers: int = 4,
    num_epochs: int = 10000,
    check_point: int = 200,
    lr: float = 0.0002,
    betas: Tuple[float] = (0.5, 0.999),
    save_root: str = "train/CNN/VGG/checkpoint/"
    ):

    device = torch.device(f"cuda:{use_gpu}" if torch.cuda.is_available() and num_gpus > 0 else "cpu")

    if img_channels == 3:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((224, 224))
        ])
    elif img_channels == 1:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            transforms.Resize((224, 224))
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

    labels_temp = train_data.class_to_idx
    labels_map = dict(zip(labels_temp.values(), labels_temp.keys()))

    if vgg_layers == 13:
        vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    elif vgg_layers == 16:
        vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    
    elif vgg_layers == 19:
        vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

    else: # vgg_layers == 11 or enter wrong number
        vgg_layers = 11
        vgg_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    model = VGG(vgg_config, in_channels=img_channels).to(device)
    model.apply(weights_init)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)

    writer = SummaryWriter(f"Tensorboard/VGG/VGG{vgg_layers}")

    for epoch in tqdm(range(0, num_epochs + 1)):
        for imgs, labels in train_loader:
            optimizer.zero_grad()

            x = imgs.to(device)
            y = labels.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            if epoch % check_point == 0:
                writer.add_scalar(f"VGG/VGG{vgg_layers}/Loss", epoch)

    writer.close()

    os.makedirs(save_root, exist_ok=True)
    torch.save(model.state_dict(), f"{save_root}/VGG{vgg_layers}/{num_epochs}_model.pth")