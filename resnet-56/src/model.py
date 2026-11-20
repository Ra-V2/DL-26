"""The CIFAR-10 ResNet-56 architecture from He et al."""

from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Two 3×3 convolutions with an option A shortcut."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        extra_channels = out_channels - in_channels
        self.channel_padding = (extra_channels // 2, extra_channels - extra_channels // 2)
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        shortcut = x[:, :, ::self.stride, ::self.stride]
        if any(self.channel_padding):
            shortcut = F.pad(shortcut, (0, 0, 0, 0, *self.channel_padding))
        return F.relu(self.residual(x) + shortcut)


def _make_stage(in_channels, out_channels, stride):
    blocks = [ResidualBlock(in_channels, out_channels, stride)]
    blocks.extend(ResidualBlock(out_channels, out_channels) for _ in range(8))
    return nn.Sequential(*blocks)


class ResNet56(nn.Module):
    """Map a batch of RGB images to ten logits per image."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            _make_stage(16, 16, stride=1),
            _make_stage(16, 32, stride=2),
            _make_stage(32, 64, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        return self.layers(x)
