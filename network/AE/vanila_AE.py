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

class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 1, out_features: int = 10):
        super().__init__()

        # in_channels x 224 x 224 -> 64x112x112
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 2, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # 64x112x112 -> 128x56x56
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 2, 2, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # 128x56x56 -> 256x28x28
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 2, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # 256x28x28 -> 512x14x14
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 2, 2, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # 512x14x14 -> 1024x7x7
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, 2, 2, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # 1024*7*7 -> 1024
        self.layer6 = nn.Sequential(
            nn.Linear(1024*7*7, 1024, bias=False),
            nn.Dropout(),
            nn.ReLU(True)
        )

        # 1024 -> out_features
        self.layer7 = nn.Sequential(
            nn.Linear(1024, out_features, bias=False),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.view(x.size(0), -1)
        x = self.layer6(x)
        x = self.layer7(x)

        return x