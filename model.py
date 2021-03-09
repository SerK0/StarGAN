from typing import Tuple

import torch
from torch import nn


class ResBlock(nn.Module):
    """
    Generator Residual block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int,
        padding: int,
    ) -> None:
        """
        :param int in_channels: input channels
        :param int out_channels: output channels
        :param Tuple[int, int] kernel_size: kernel size
        :param int stride: stride of filter
        :param int padding: padding size
        :returns: None
        """
        super(ResBlock, self).__init__()

        self.residual_connection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor x: input tensor
        :rtype: torch.Tensor
        """
        return self.residual_connection(x) + x


class ConvBlock(nn.Module):
    """
    Downsampling block
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int,
        padding: int,
        direction: str = None,
    ) -> None:
        """
        :param int in_channels: number of input channels
        :param int out_channels: number of output channels
        :param Tuple[int, int] kernel_size: kernel size
        :param int stride: stride of filter
        :param int padding: padding size
        :param str direction: 'downsample' or 'upsample'
        """
        super(ConvBlock, self).__init__()

        assert direction in [
            "upsample",
            "downsample",
        ], "direction must be [upsample] or [downsample]"

        if direction == "downsample":
            conv_class = nn.Conv2d
        else:
            conv_class = nn.ConvTranspose2d

        self.pipeline = nn.Sequential(
            conv_class(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor x: image of size B x C x H x W
        :rtype: torch.Tensor
        """
        return self.pipeline(x)


class Generator(nn.Module):
    """
    Generator
    """

    def __init__(self, in_channels):

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


if __name__ == "__main__":
    image = torch.rand(4, 3, 128, 128)
    model = Generator(in_channels=3)

    print(model(image).size())
