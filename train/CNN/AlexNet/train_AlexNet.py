from network.CNN.AlexNet import AlexNet
from network.weights_init import weights_init

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple

# ------------------------------------------------------------------- #

def train_AlexNet(
    num_gpus: int = 3,
    use_gpu: int = 1,
    batch_size: int = 7300,
    img_channels: int = 3,
    num_workers: int = 4,
    num_epochs: int = 10000,
    check_point: int = 200,
    lr: float = 0.0002,
    save_root: str = "train/CNN/AlexNet/checkpoint/"
    ):
    device = torch.device(f"cuda:{use_gpu}" if torch.cuda.is_available() and num_gpus > 0 else "cpu")

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
        num_workers=num_workers,
    )

    labels_temp = train_data.class_to_idx
    labels_map = dict(zip(labels_temp.values(), labels_temp.keys()))

    model = AlexNet(in_channels=1, out_features=10).to(device)
    model.apply(weights_init)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    writer = SummaryWriter("Tensorboard/AlexNet")

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
                writer.add_scalar("AlexNet/Loss", loss.item(), epoch)

    writer.close()
    os.makedirs(save_root, exist_ok=True)
    torch.save(model.state_dict(), f"{save_root}/{num_epochs}_model.pth")