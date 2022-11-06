import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_features: int = 10):
        super().__init__()

        # in_channelsx227x227 -> 96x27x27
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, 11, 4, 0),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 0)
        )

        # 96x27x27 -> 256x13x13
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 0)
        )

        # 256x13x13 -> 384x13x13
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(True)
        )

        # 384x13x13 -> 384x13x13
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(True)
        )

        # 384x13x13 -> 256x6x6
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 0)
        )

        # 256*6*6 -> 4096
        self.layer6 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(True)
        )

        # 4096 -> 4096
        self.layer7 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True)
        )

        # 4096 -> out_features
        self.layer8 = nn.Sequential(
            nn.Linear(4096, out_features=out_features),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = x.view(-1, 256*6*6)

        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        return x