from network.CNN.LeNet5 import LeNet5

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

# --------------------------------------------------------------- #

def train_LeNet5(
    num_gpus: int = 3,
    batch_size: int = 15000,
    num_workers: int = 4,
    num_epochs: int = 5000,
    check_point: int = 200,
    lr: float = 0.0002,
    betas: Tuple[float] = (0.5, 0.999)
):
    ##################
    # Hyperparameter #
    ##################

    device = torch.device("cuda" if torch.cuda.is_available() and num_gpus > 0 else "cpu")

