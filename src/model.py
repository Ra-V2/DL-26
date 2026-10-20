import torch
import torch.nn as nn

from .highway import ConvHighwayLayer


class HighwayCNN(nn.Module):
    def __init__(
        self,
        num_classes=10,
        channels=32,
        num_highway_layers=4,
        gate_bias=-2.0,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.highway_layers = nn.Sequential(
            *[
                ConvHighwayLayer(
                    channels=channels,
                    gate_bias=gate_bias,
                )
                for _ in range(num_highway_layers)
            ]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.highway_layers(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x


class StandardCNN(nn.Module):
    def __init__(
        self,
        num_classes=10,
        channels=32,
        num_layers=4,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.conv_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                )
                for _ in range(num_layers)
            ]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv_layers(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)