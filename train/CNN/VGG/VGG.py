import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, in_channels: int = 3, out_features: int = 10):
        super().__init__()

        
    def forward(self, x):
        return x