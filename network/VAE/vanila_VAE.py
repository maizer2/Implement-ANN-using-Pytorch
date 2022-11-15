import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        

    def forward(self, x):

        return x

class Decoder(nn.Module):
    def __init__(self, out_channels: int = 1):
        super().__init__()

    def forward(self, x):
        return x

class VAE(nn.Module):
    def __init__(self, img_channels: int = 1):
        super().__init__()
        
        self.encoder = Encoder(img_channels)
        self.decoder = Decoder(img_channels)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x