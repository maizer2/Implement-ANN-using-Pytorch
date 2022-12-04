import torch
import torch.nn as nn

from typing import Optional, List

# ------------------------------------------------------- #

class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        condition_dim: int = 10,
        hidden_channels: Optional[List] = None,
        out_channels: int = 1
        ):
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [1024, 512, 256, 128, 64]

        layers = []

        in_channels = latent_dim + condition_dim

        for h_channels in hidden_channels:
            layers += [
                nn.ConvTranspose2d(in_channels, h_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(h_channels),
                nn.ReLU(True)
            ]

            if h_channels == hidden_channels[-1]:
                layers += [
                    nn.ConvTranspose2d(h_channels, out_channels, 4, 2, 1, bias=True),
                    nn.Tanh()
                ]
            else:
                in_channels = h_channels
        
        self.model = nn.Sequential(*layers)

    def forward(self, z, condition):
        z = torch.cat((z, condition), 1).unsqueeze(-1).unsqueeze(-1)
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: Optional[List] = None,
        ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [64, 128, 256, 512, 1024]

        layers = []

        for h_channels in hidden_channels:
            layers += [
                nn.Conv2d(in_channels, h_channels, 2, 2, 0, bias=False),
                nn.BatchNorm2d(h_channels),
                nn.LeakyReLU(0.2, True)
            ]

            if h_channels == hidden_channels[-1]:
                layers += [
                    nn.Conv2d(h_channels, 1, 2, 1, 0, bias=False),
                    nn.Sigmoid()
                ]
            else:
                in_channels = h_channels

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).clone().view(-1)
