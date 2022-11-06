import torch.nn as nn
from typing import List

class vgg_layers(nn.Module):
    def __init__(
        self, 
        config: List[str], 
        normalize: bool =  True,
        in_channels: int = 3
        ):

        super().__init__()

        layers = []

        for out_channels in config:
            assert out_channels == "M" or isinstance(out_channels, int)

            if out_channels == "M":
                layers.append(
                    nn.MaxPool2d(2, 1, 1)
                )
            else:
                layers.append(
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1)
                )
                if normalize:
                    layers.append(
                        nn.BatchNorm2d(out_channels)
                    )
                layers.append(
                    nn.ReLU(True)
                )
                in_channels = out_channels

            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)