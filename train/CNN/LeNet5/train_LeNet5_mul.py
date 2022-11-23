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

# Multi GPU
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# --------------------------------------------------------------- #

def train_LeNet5_mul(
    batch_size: int = 2**5,
    img_channels: int = 1,
    num_workers: int = 6,
    num_epochs: int = 5000,
    check_point: int = 20,
    lr: float = 0.001,
    save_root: str = "train/CNN/LeNet5/checkpoint/"
):
    ##################
    # Hyperparameter #
    ##################

    dist.init_process_group("nccl")
    os.makedirs(save_root, exist_ok=True)
    gpu_id = int(os.environ["LOCAL_RANK"])

    ###########
    # Prepare #
    ###########

    if img_channels == 3:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((32, 32))
        ])
    elif img_channels == 1:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            transforms.Resize((32, 32))
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
        sampler=data.DistributedSampler(train_data),
        num_workers=num_workers,
        pin_memory=True
    )

    labels_temp = train_data.class_to_idx
    labels_map = dict(zip(labels_temp.values(), labels_temp.keys()))

    model = LeNet5().to(gpu_id)
    model = DDP(model, [gpu_id])
    model.apply(weights_init)

    criterion = nn.CrossEntropyLoss().to(gpu_id)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    #########
    # Train #
    #########
    
    writer = SummaryWriter("Tensorboard/CNN/LeNet5_mul")

    for epoch in tqdm(range(0, num_epochs + 1)):
        train_loader.sampler.set_epoch(epoch)
        for imgs, labels in train_loader:
            optimizer.zero_grad()

            ###########
            # Prepard #
            ###########

            x = imgs.to(gpu_id)
            y = labels.to(gpu_id)

            ############
            # Training #
            ############

            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            if epoch % check_point == 0:
                writer.add_scalar(f"Loss/LeNet5/batch:{batch_size}, lr:{lr}", loss.item(), epoch)
                torch.save(model.state_dict(), f"{save_root}/{epoch}_model.pth")

    writer.close()