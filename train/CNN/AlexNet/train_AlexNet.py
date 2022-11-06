from network.CNN.AlexNet import AlexNet
from network.weights_init import weights_init

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt

from tqdm import tqdm
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple

# ------------------------------------------------------------------- #

def train_AlexNet(
    num_gpus: int = 3,
    batch_size: int = 7300,
    num_workers: int = 4,
    num_epochs: int = 5000,
    check_point: int = 200,
    lr: float = 0.0002
):
    device = torch.device("cuda:1" if torch.cuda.is_available() and num_gpus > 0 else "cpu")

    train_data = datasets.MNIST(
        root="/data/DataSet/",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            transforms.Resize((227, 227))
        ])
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    writer = SummaryWriter("Tensorboard/AlexNet")

    for epochs in tqdm(range(0, num_epochs + 1)):
        for imgs, labels in train_loader:
            optimizer.zero_grad()

            x = imgs.to(device)
            y = labels.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            if epochs % check_point == 0:
                writer.add_scalar("AlexNet/Loss", loss.item(), epochs)

    writer.close()
    torch.save(model.state_dict(), "train/CNN/AlexNet/checkpoint/model.pth")