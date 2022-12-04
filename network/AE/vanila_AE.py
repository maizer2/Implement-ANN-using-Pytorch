import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_space_dim: int = 4):
        super().__init__()

        self.conv = nn.Sequential(
            # in_channels x 28 x 28 -> 8x14x14
            nn.Conv2d(in_channels, 8, 3, 2, 1, bias=False),
            nn.ReLU(True),
            # 8x14x14 -> 16x7x7
            nn.Conv2d(8, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 16x7x7 -> 32x3x3
            nn.Conv2d(16, 32, 3, 2, 0, bias=False),
            nn.ReLU(True),
        )

        self.linear = nn.Sequential(
            # 32*3*3 -> 128
            nn.Linear(32*3*3, 128, bias=False),
            nn.ReLU(True),
            nn.Linear(128, latent_space_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

class Decoder(nn.Module):
    def __init__(self, out_channels: int = 1, latent_space_dim: int = 4):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(latent_space_dim, 128, bias=False),
            nn.ReLU(True),
            nn.Linear(128, 32*3*3),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 2, 0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 8, 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.ConvTranspose2d(8, out_channels, 3, 2, 1, output_padding=1, bias=False)
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 32, 3, 3)
        x = self.deconv(x)

        return x

class AutoEncoder(nn.Module):
    def __init__(self, img_channels: int = 1):
        super().__init__()
        
        self.encoder = Encoder(img_channels)
        self.decoder = Decoder(img_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x