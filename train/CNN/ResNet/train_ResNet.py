from network.CNN.ResNet import ResNet, BuildingBlock, Bottleneck
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
from tqdm import tqdm
from typing import Tuple, NamedTuple
from collections import namedtuple

# ---------------------------------------------------------------- #

def train_ResNet(
    num_gpus: int = 3,
    use_gpu: int = 0,
    batch_size: int = 500,
    img_channels: int = 3,
    layers:int = 18, # 34, 50, 101, 152
    num_workers: int = 4,
    num_epochs: int = 10000,
    check_point: int = 200,
    lr: float = 1e-04,
    betas: Tuple[float] = (0.5, 0.999),
    save_root: str = "train/CNN/ResNet/checkpoint/"
    ):

    os.makedirs(f"{save_root}/ResNet{layers}", exist_ok=True)
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

    resnet_config = namedtuple("resnet_config", ["block", "n_blocks", "channels"])
    if layers == 34:
        config = resnet_config(
            block=BuildingBlock,
            n_blocks=[3, 4, 6, 3],
            channels=[64, 128, 256, 512])

    elif layers == 50:
        config = resnet_config(
            block=BuildingBlock,
            n_blocks=[3, 4, 6, 3],
            channels=[64, 128, 256, 512])
    elif layers == 101:
        config = resnet_config(
            block=Bottleneck,
            n_blocks=[3, 4, 23, 3],
            channels=[64, 128, 256, 512])
    elif layers == 152:
        config = resnet_config(
            block=Bottleneck,
            n_blocks=[3, 8, 36, 3],
            channels=[64, 128, 256, 512])
    else: # layers == 18 or enter wrong number
        layers = 18
        config = resnet_config(
            block=Bottleneck,
            n_blocks=[2, 2, 2, 2],
            channels=[64, 128, 256, 512])

    model = ResNet(config, in_channels=img_channels, out_features=len(labels_map)).to(device)
    model.apply(weights_init)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)

    writer = SummaryWriter(f"Tensorboard/ResNet/ResNet{layers}")

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
                writer.add_scalar(f"Loss/ResNet/ResNet{layers}", loss.item(), epoch)
                torch.save(model.state_dict(), f"{save_root}/ResNet{layers}/{epoch}_model.pth")

    writer.close()