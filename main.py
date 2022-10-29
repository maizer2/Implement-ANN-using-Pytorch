from network.GAN.dcgan import Generator, Discriminator

import torch
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tqdm import tqdm

# --------------------------------------------------------------- #

batch_size = 1
num_workers = 4

if __name__ == "__main__":

    train_data = datasets.MNIST(
        root="/data/DataSet/",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
    )

    train_loader = data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,

    )

    num_epochs = 20000

    for epochs in tqdm(range(1, num_epochs + 1)):
        for idx, (imgs, _) in enumerate(train_loader):
            