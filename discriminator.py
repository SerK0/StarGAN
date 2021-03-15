from typing import Tuple

import torch
from torch import nn

from common.blocks import DiscriminatorConvBlock


class Discriminator(nn.Module):
    """
    Discriminator
    """

    def __init__(
        self, height: int, width: int, n_domain: int, in_channels: int = 3
    ) -> None:
        """
        :param int height: height of input image
        :param int width: width of input image
        :param int n_domain: how many domains classify
        :param int in_channels: number of channels in image
        :rtype: None
        """
        super(Discriminator, self).__init__()

        self.input_layer = DiscriminatorConvBlock(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(4, 4),
            stride=2,
            padding=1,
        )

        self.hidden_layers = nn.Sequential(
            DiscriminatorConvBlock(
                in_channels=64,
                out_channels=128,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            DiscriminatorConvBlock(
                in_channels=128,
                out_channels=256,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            DiscriminatorConvBlock(
                in_channels=256,
                out_channels=512,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            DiscriminatorConvBlock(
                in_channels=512,
                out_channels=1024,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
            DiscriminatorConvBlock(
                in_channels=1024,
                out_channels=2048,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),
        )

        self.output_src = nn.Conv2d(
            2048, 1, kernel_size=(3, 3), stride=1, padding=1, bias=False
        )
        self.output_cls = nn.Conv2d(
            2048,
            n_domain,
            kernel_size=(height // 64, width // 64),
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param torch.Tensor x: image with dims B x C x H x W
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        :returns: Feature map(Real or Fake) and distribution of domain labels
        output_src[batch_size, 1, h/64, w/64]
        output_cls[batch_size, n_domain, 1, 1]
        """
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        output_src = self.output_src(x)
        output_cls = self.output_cls(x)
        return output_src, output_cls
