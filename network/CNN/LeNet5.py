import torch.nn as nn

# -----------------------------------------------

'''
Input 1x32x32
layer1 1x32x32 -> 6x28x28
layer2 6x28x28 -> 6x14x14
layer3 6x14x14 -> 16x10x10
layer4 16x10x10 -> 16x5x5
To Linear 16x5x5 -> 16*5*5(450)
layer5 450 -> 120
layer6 120 -> 84
layer7 84 -> 10 (Classification)
'''

class LeNet5(nn.Moduel):
    def __init__(self):
        super().__init__()

        # 1x32x32 -> 6x28x28
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 0, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(True)
        )
        # 6x28x28 -> 6x14x14
        self.layer2 = nn.Sequential(
            nn.AvgPool2d(2)
        )

        # 6x14x14 -> 16x10x10
        self.layer3 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        # 16x10x10 -> 16x5x5
        self.layer4 = nn.Sequential(
            nn.AvgPool2d(2)
        )

        # 16*5*5 -> 120
        self.layer5 = nn.Sequential(
            nn.Linear(16*5*5, 120, bias=False),
            nn.BatchNorm2d(120),
            nn.ReLU(True)
        )

        # 120 -> 84
        self.layer6 = nn.Sequential(
            nn.Linear(120, 84, bias=False),
            nn.BatchNorm2d(84),
            nn.ReLU(True)
        )

        # 84 -> 10
        self.layer7 = nn.Sequential(
            nn.Linear(84, 10, bias=False),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(-1, 16*5*5)

        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        
        return x