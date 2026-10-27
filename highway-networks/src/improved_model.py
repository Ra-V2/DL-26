"""Final multi-stage CNNs used in notebooks 05 and 06.

Module names and structure match the saved checkpoints in
checkpoints/improved_highway/ and checkpoints/final_comparison/, so do not
rename attributes (stem, stage1, down1, ..., head) without migrating them.
"""

import torch
import torch.nn as nn


HIGHWAY_CONFIG = dict(
    num_classes=10,
    widths=(32, 64, 128),
    blocks_per_stage=2,
    gate_bias=-3.0,
    dropout=0.2,
)

STANDARD_CONFIG = dict(
    num_classes=10,
    widths=(32, 64, 128),
    blocks_per_stage=2,
    dropout=0.2,
)


class ImprovedHighwayBlock(nn.Module):
    """y = H(x) * T(x) + x * (1 - T(x)), with H = ReLU(BN(Conv(x)))."""

    def __init__(self, channels, gate_bias=-3.0):
        super().__init__()

        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )
        self.gate = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, x):
        transformed = self.transform(x)
        t = torch.sigmoid(self.gate(x))

        return transformed * t + x * (1.0 - t)


class StandardBlock(nn.Module):
    """Plain Conv-BN-ReLU block with the same transform as the Highway block."""

    def __init__(self, channels):
        super().__init__()

        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.transform(x)


def _conv_bn_relu(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class _MultiStageCNN(nn.Module):
    """Shared backbone: stem -> 3 stages with 2 stride-2 downsamples -> head.

    Subclasses only decide which block fills each stage.
    """

    def __init__(self, make_block, num_classes, widths, blocks_per_stage, dropout):
        super().__init__()
        w1, w2, w3 = widths

        self.stem = _conv_bn_relu(3, w1)
        self.stage1 = nn.Sequential(
            *[make_block(w1) for _ in range(blocks_per_stage)]
        )
        self.down1 = _conv_bn_relu(w1, w2, stride=2)
        self.stage2 = nn.Sequential(
            *[make_block(w2) for _ in range(blocks_per_stage)]
        )
        self.down2 = _conv_bn_relu(w2, w3, stride=2)
        self.stage3 = nn.Sequential(
            *[make_block(w3) for _ in range(blocks_per_stage)]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(w3, num_classes),
        )

    def forward(self, x):
        x = self.stage1(self.stem(x))
        x = self.stage2(self.down1(x))
        x = self.stage3(self.down2(x))

        return self.head(x)


class ImprovedHighwayCNN(_MultiStageCNN):
    def __init__(
        self,
        num_classes=10,
        widths=(32, 64, 128),
        blocks_per_stage=2,
        gate_bias=-3.0,
        dropout=0.2,
    ):
        super().__init__(
            lambda channels: ImprovedHighwayBlock(channels, gate_bias),
            num_classes,
            widths,
            blocks_per_stage,
            dropout,
        )


class ImprovedStandardCNN(_MultiStageCNN):
    def __init__(
        self,
        num_classes=10,
        widths=(32, 64, 128),
        blocks_per_stage=2,
        dropout=0.2,
    ):
        super().__init__(
            StandardBlock,
            num_classes,
            widths,
            blocks_per_stage,
            dropout,
        )


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
