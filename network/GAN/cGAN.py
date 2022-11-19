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