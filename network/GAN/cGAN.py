import torch
import torch.nn as nn

from typing import Optional, List

# ------------------------------------------------------- #

class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        hidden_dim: Optional[List] = None,
        out_channels: int = 1
        ):
        super().__init__()



    def forward(self, z, label):
        z = torch.cat((z, label), 1)
        return z

class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: Optional[List] = None,
        out_features: int = 10,
        ):
        super().__init__()
    
    def forward(self, x):
        return x
