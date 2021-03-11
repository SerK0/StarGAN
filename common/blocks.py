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
        :rtype: None
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


class DiscriminatorConvBlock(nn.Module):
    """
    Discriminator Convolution block
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int,
        padding: int,
        negative_slope: float = 0.01,
    ) -> None:
        """
        :param int in_channels: number of input channels
        :param int out_channels: number of output channels
        :param Tuple[int, int] kernel_size: kernel size
        :param int stride: stride of filter
        :param int padding: padding size
        :rtype: None
        """
        super(DiscriminatorConvBlock, self).__init__()

        self.pipeline = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.LeakyReLU(negative_slope=negative_slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor x: image of size B x C x H x W
        :rtype: torch.Tensor
        """
        return self.pipeline(x)
