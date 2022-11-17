import torch
import torch.nn as nn

from typing import Optional, List

class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        layers = []

        for h_dim in hidden_dims:
            layers += [
                nn.Conv2d(in_channels, h_dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2, True)
            ]
            in_channels = h_dim

        self.encoder = nn.Sequential(*layers)
        self.mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.var = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        # 1x28x28 -> 512x1x1
        x = self.encoder(x)
        # 512
        x = x.view(x.size(0), -1)
        # 512 -> latent_dim
        mean, var  = self.mean(x), self.var(x)

        return mean, var

class Decoder(nn.Module):
    def __init__(self, out_channels: int, latent_dim: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]
        else:
            if hidden_dims[0] < hidden_dims[-1]:
                hidden_dims.reverse()

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0], bias=False),
            nn.LeakyReLU(0.2, True)
        )

        layers = []

        for idx in range(len(hidden_dims)):

            if hidden_dims[idx] != hidden_dims[-1]:
                layers += [
                    nn.ConvTranspose2d(hidden_dims[idx], hidden_dims[idx + 1], 3, 2, 1, 1, bias=False),
                    nn.BatchNorm2d(hidden_dims[idx + 1]),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                layers += [
                    nn.ConvTranspose2d(hidden_dims[idx], out_channels, 2, 2, 2, bias=False),
                    nn.Sigmoid()
                ]
                
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 512, 1, 1)
        x = self.decoder(x)
        return x

class vanilaVAE(nn.Module):
    def __init__(self, img_channels: int = 1, latent_dim: int = 200, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        
        self.encoder = Encoder(img_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(img_channels, latent_dim, hidden_dims)

    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        
        return std * epsilon + mean

    def forward(self, x):
        mean, var = self.encoder(x)
        z = self.reparameterization(mean, var)
        x_hat = self.decoder(z)
        return x_hat, mean, var

def KLD_loss(mean, log_var):
    return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
