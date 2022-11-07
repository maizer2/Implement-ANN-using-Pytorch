from network.CNN.LeNet5 import LeNet5
from network.weights_init import weights_init

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

def train_LeNet5(
    num_gpus: int = 3,
    use_gpu: int = 0,
    batch_size: int = 60000,
    num_workers: int = 4,
    num_epochs: int = 10000,
    check_point: int = 200,
    lr: float = 0.0002,
    save_root: str = "train/CNN/LeNet5/checkpoint/"
):
    ##################
    # Hyperparameter #
    ##################

    device = torch.device(f"cuda:{use_gpu}" if torch.cuda.is_available() and num_gpus > 0 else "cpu")

    ###########
    # Prepare #
    ###########

    train_data = datasets.MNIST(
        root="/data/DataSet/",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            transforms.Resize((32, 32))
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

    model = LeNet5().to(device)
    model.apply(weights_init)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    #########
    # Train #
    #########
    
    writer = SummaryWriter("Tensorboard/LeNet5")

    for epochs in tqdm(range(0, num_epochs + 1)):
        for imgs, labels in train_loader:
            optimizer.zero_grad()

            ###########
            # Prepard #
            ###########

            x = imgs.to(device)
            y = labels.to(device)

            ############
            # Training #
            ############

            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            if epochs % check_point == 0:
                writer.add_scalar("LeNet5/Loss", loss.item(), epochs)
    
    writer.close()
    os.makedirs(save_root, exist_ok=True)
    torch.save(model.state_dict(), f"{save_root}/{num_epochs}_model.pth")
