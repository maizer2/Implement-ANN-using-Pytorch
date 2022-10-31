import torch.nn as nn

'''
Using MNIST 1x28x28 DataSet
'''

# --------------------------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# --------------------------------------------------------------- #

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # 100x1x1 -> 1024x2x2
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, 2, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )
        # 1024x2x2 -> 512x4x4
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        # 512x4x4 -> 256x8x8
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        # 256x8x8 -> 128x16x16
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        # 128x16x16 -> 1x28x28
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 2, 2, 2, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

# --------------------------------------------------------------- #

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1x28x28 -> 64x16x16
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 2, 2, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True)
        )
        # 64x16x16 -> 128x8x8
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True)
        )
        # 128x8x8 -> 256x4x4
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True)
        )
        # 256x4x4 -> 512x2x2
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True)
        )
        # 512x2x2 -> 1x1x1
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
