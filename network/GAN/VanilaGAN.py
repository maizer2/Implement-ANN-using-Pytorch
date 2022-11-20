import torch.nn as nn

from typing import Union, List, Optional

'''
Using MNIST 1x28x28 DataSet
'''

# --------------------------------------------------------------------- #

# latent_vector -> out_features
class Generator(nn.Module):
    def __init__(
        self, 
        latent_dim: int = 100, 
        hidden_features: Optional[List[int]] = None,
        out_features: int = 1*28*28,
        img_channels: int = 1
        ):
        super().__init__()

        self.img_channels = img_channels
        self.out_features = out_features

        if hidden_features is None:
            hidden_features = [256, 512, 1024]

        layers = []

        in_features = latent_dim

        for h_feature in hidden_features:
            layers += [
                nn.Linear(in_features, h_feature, bias=False),
                nn.LeakyReLU(0.02, True)
            ]

            if h_feature == hidden_features[-1]:
                layers += [
                nn.Linear(h_feature, out_features),
                nn.Softmax(1)
                ]
            else:
                in_features = h_feature

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        z = self.model(z)
        z = z.view(-1, self.img_channels, int((self.out_features**(1/2))/self.img_channels), int((self.out_features**(1/2))/self.img_channels))
        return z

class Discriminator(nn.Module):
    def __init__(
        self,
        in_features: int = 784,
        hidden_features: Optional[List[int]] = None,
        out_features: int = 1,
        ):
        super().__init__()

        if hidden_features is None:
            hidden_features = [64, 128, 256, 512]

        layers = []

        for h_feature in hidden_features:
            layers += [
                nn.Linear(in_features, h_feature, bias=False),
                nn.Dropout(0.3)
            ]

            if h_feature == hidden_features[-1]:
                layers += [
                    nn.Linear(h_feature, out_features, bias=False),
                    nn.Softmax(1)
                ]
            else:
                in_features = h_feature

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x).view(-1)