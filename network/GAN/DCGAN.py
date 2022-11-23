import torch.nn as nn

from typing import Optional, List
'''
Using MNIST 1x28x28 DataSet
'''

# --------------------------------------------------------------- #

class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        hidden_channels: Optional[List[int]] = None,
        img_channels: int = 1
        ):
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [1024, 512, 256, 128, 64]

        layers = []

        in_channels = latent_dim

        for h_channel in hidden_channels:
            layers += [
                nn.ConvTranspose2d(in_channels, h_channel, 4, 2, 1, bias=False),
                nn.BatchNorm2d(h_channel),
                nn.ReLU(True)
            ]

            if h_channel == hidden_channels[-1]:
                layers += [
                    nn.ConvTranspose2d(h_channel, img_channels, 4, 2, 1, bias=False),
                    nn.Tanh()
                ]
            else:
                in_channels = h_channel

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)

# --------------------------------------------------------------- #

class Discriminator(nn.Module):
    def __init__(
        self,
        in_channel: int = 1,
        hidden_channels: Optional[List[int]] = None,
        out_channels: int = 1
        ):
        super().__init__()
        # 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
        if hidden_channels is None:
            hidden_channels = [64, 128, 256, 512, 1024]

        layers = []

        for h_channel in hidden_channels:
            layers += [
                nn.Conv2d(in_channel, h_channel, 2, 2, 0, bias=False),
                nn.BatchNorm2d(h_channel),
                nn.LeakyReLU(0.2, True)
            ]

            if h_channel == hidden_channels[-1]:
                layers += [
                    nn.Conv2d(h_channel, out_channels, 2, 1, 0, bias=False),
                    nn.Sigmoid()
                ]
            else:
                in_channel = h_channel

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).clone().view(-1)
