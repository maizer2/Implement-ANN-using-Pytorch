import torch.nn as nn
from typing import NamedTuple

class BuildingBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride=1, downsample=False):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),

            nn.ReLU(True),

            nn.Conv2d(out_channels, self.expansion*out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.expansion*out_channels)
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            identitiy = self.downsample(x)
        else:
            identitiy = x

        x = self.layer(x)
        
        return nn.ReLU(True)(x + identitiy)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride=1, downsample=False):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            
            nn.ReLU(True),

            nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),

            nn.ReLU(True),

            nn.Conv2d(out_channels, self.expansion*out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.expansion*out_channels)
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            identitiy = self.downsample(x)
        else:
            identitiy = x

        x = self.layer(x)
        
        return nn.ReLU(True)(x + identitiy)

class ResNet(nn.Module):
    def __init__(self, config: NamedTuple, in_channels: int, out_features: int):
        super().__init__()

        block, n_blocks, channels = config
        self.in_channels = channels[0]
        
        assert len(n_blocks) == len(channels) == 4

        # in_channels x 224 x 224 -> 64x112x112 -> 64x56x56
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, self.in_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1)
        )
        # 64x56x56 -> 64x56x56
        self.layer2 = self.get_resnet_layer(block, n_blocks[0], channels[0])

        # 64x56x56 -> 128x28x28
        self.layer3 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)

        # 128x28x28 -> 256x14x14
        self.layer4 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)

        # 256x14x14 -> 512x7x7
        self.layer5 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.layer6 = nn.Sequential(
            nn.Linear(self.in_channels, out_features),
            nn.Softmax(1)
            )

    def get_resnet_layer(self, block, n_blocks, channels, stride=1):
        layers = []

        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion*channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.layer6(x)
        return x