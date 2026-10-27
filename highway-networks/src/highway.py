import torch
import torch.nn as nn


class ConvHighwayLayer(nn.Module):
    def __init__(self, channels, gate_bias=-2.0):
        super().__init__()

        self.transform = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1
        )

        self.gate = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1
        )

        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, x):
        H = torch.relu(self.transform(x))
        T = torch.sigmoid(self.gate(x))

        return H * T + x * (1 - T)