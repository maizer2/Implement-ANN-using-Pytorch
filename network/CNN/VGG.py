import torch.nn as nn

from typing import List,Union

class VGG(nn.Module):
    def __init__(
        self, 
        config: List[Union[int, str]],
        batch_norm: bool = True,
        in_channels: int = 1,
        out_features: int = 10
        ):

        super().__init__()
        
        def feature_extract(config=config, batch_norm=batch_norm, in_channels=in_channels):
            layers = []

            for out_channels in config:
                assert out_channels == "M" or isinstance(out_channels, int)

                if out_channels == "M":
                    layers.append(
                        # nn.MaxPool2d(2, 1, 1)
                        nn.MaxPool2d(2)
                    )
                else:
                    layers.append(
                        nn.Conv2d(in_channels, out_channels, 3, 1, 1)
                    )
                    if batch_norm:
                        layers.append(
                            nn.BatchNorm2d(out_channels)
                        )
                    layers.append(
                        nn.ReLU(True)
                    )
                    in_channels = out_channels

            return nn.Sequential(*layers)

        # 512*7*7 -> 4096 -> out_features
        def classifier(out_features: int = 10):
            layers = nn.Sequential(
                nn.Linear(512*7*7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, out_features),
                nn.Softmax(1)
            )

            return layers

        self.features = feature_extract(config, batch_norm, in_channels)

        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.classifier = classifier(out_features)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, 512*7*7)
        x = self.classifier(x)
        return x