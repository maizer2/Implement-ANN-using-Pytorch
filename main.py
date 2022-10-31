from network.GAN.dcgan import Generator, Discriminator, weights_init

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tqdm import tqdm

# --------------------------------------------------------------- #


if __name__ == "__main__":

    ##################
    # Hyperparameter #
    ##################

    num_gpus = 3
    batch_size = 128
    num_workers = 4
    num_epochs = 20000
    device = torch.device("cuda" if torch.cuda.is_available() and num_gpus > 0 else "cpu")
    latent_space_vector = 100
    lr = 0.0002
    betas = (0.5, 0.999)

    ###########
    # Prepare #
    ###########

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

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    if (device.type == "cuda") and (num_gpus > 1):
        netG = nn.DataParallel(netG, list(range(num_gpus)))
        netD = nn.DataParallel(netD, list(range(num_gpus)))

    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerG = optim.Adam(
        params=netG.parameters(),
        lr=lr,
        betas=betas
    )
    optimizerD = optim.Adam(
        params=netD.parameters(),
        lr=lr,
        betas=betas
    )
    #########
    # Train #
    #########

    for epochs in tqdm(range(1, num_epochs + 1)):
        for idx, (imgs, _) in enumerate(train_loader):
            netG.zero_grad(), netD.zero_grad()

            x = imgs.to(device)
            y = torch.ones((x.size(0), ), dtype=torch.float, device=device)
            noise = torch.randn((x.size(0), latent_space_vector, 1, 1), dtype=torch.float, device=device)

            ##########################
            # Training Discriminator #
            ##########################

            y_hat = netD(x).view(-1)
            loss = criterion(y_hat, y)
            
            print(loss)

            
            break
        break