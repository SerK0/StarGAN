from typing import Tuple

import torch
from torch import nn

from common.blocks import ConvBlock, ResBlock


class Generator(nn.Module):
    """
    Generator
    """

    def __init__(self, in_channels) -> None:
        """
        :param int in_channels: number of input channels
        :rtype: None
        """
        super(Generator, self).__init__()

        self.downsampling = nn.Sequential(
            ConvBlock(
                in_channels,
                64,
                kernel_size=(7, 7),
                stride=1,
                padding=3,
                direction="downsample",
            ),
            ConvBlock(
                64,
                128,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                direction="downsample",
            ),
            ConvBlock(
                128,
                256,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                direction="downsample",
            ),
        )

        self.bottleneck = nn.Sequential(
            ResBlock(256, 256, (3, 3), 1, 1),
            ResBlock(256, 256, (3, 3), 1, 1),
            ResBlock(256, 256, (3, 3), 1, 1),
            ResBlock(256, 256, (3, 3), 1, 1),
            ResBlock(256, 256, (3, 3), 1, 1),
            ResBlock(256, 256, (3, 3), 1, 1),
        )

        self.upsampling = nn.Sequential(
            ConvBlock(
                256,
                128,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                direction="upsample",
            ),
            ConvBlock(
                128,
                64,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                direction="upsample",
            ),
        )

        self.conv_final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=(7, 7), stride=1, padding=3), nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        """
        :param torch.Tensor x: image with dims B x C x H x W
        :rtype: torch.Tensor
        """
        x = self.downsampling(x)
        x = self.bottleneck(x)
        x = self.upsampling(x)
        return self.conv_final(x)
