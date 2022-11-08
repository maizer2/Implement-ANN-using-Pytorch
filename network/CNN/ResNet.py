import torch.nn as nn
from typing import *

class ResNet(nn.Module):
    def __init__(self):
        def ResBlock(in_channels: int):
            layer = nn.Sequential(
                # I +2p -3 / S = O
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),

                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True)
            )

            layer = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),

                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),

                nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True)
            )
    def forward(self, x):
        return x