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

        self.output_src = nn.Conv2d(2048, 1, kernel_size=(3, 3), stride=1, padding=1)
        self.output_cls = nn.Conv2d(
            2048, n_domain, kernel_size=(height // 64, width // 64), stride=1, padding=0
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


class StarGan(nn.Module):
    def __init__(
        self,
        height: int = 64,
        width: int = 64,
        n_domain: int = 5,
        in_channels: int = 3,
        device="cpu",
    ) -> None:

        super(StarGan, self).__init__()

        self.height = height
        self.width = width
        self.G = Generator(in_channels=in_channels + n_domain)
        self.D = Discriminator(height, width, n_domain, in_channels)

        self.to(device)

    def forward(self, x):
        pass

    def to(self, device):
        self.D.to(device)
        self.G.to(device)

    def train(self):
        self.G.train()
        self.D.train()

    def eval(self):
        self.G.eval()
        self.D.eval()

    def concat_image_label(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        """
        :param torch.Tensor image: size batch_size x 3 x height x width
        :param torch.Tensor label: size batch_size x n_domain
        :rtype: torch.Tensor
        :returns: concatenated image and labels of size batch_size x (3 + n_domain) x height x width
        """
        label = label[:, :, None, None].repeat(1, 1, self.height, self.width)
        return torch.cat((image, label), dim=1)

    def trainG(self, image, label):
        pass

    def trainD(self, image, label):
        pass

    @torch.no_grad()
    def generate(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor image: size batch_size x 3 x height x width
        :param torch.Tensor label: size batch_size x n_domain
        :rtype: torch.Tensor
        :returns: images of size batch_size x 3 x height x width
        """
        self.eval()
        images_labels = self.concat_image_label(image, label)
        return self.G(images_labels)


if __name__ == "__main__":
    from imageio import imsave
    from loss import (
        AdversarialLoss,
        DomainClassificationLoss,
        ReconstructionLoss,
        GeneratorLoss,
    )

    batch_size, channels, n_domain, height, width = (4, 3, 5, 128, 128)
    images = torch.rand(batch_size, channels, height, width)
    labels = torch.rand(batch_size, n_domain)

    model = StarGan(height, width)
    model.train()
    fake = model.G(model.concat_image_label(images, labels))
    model.D.eval()
    output_src, output_cls = model.D(fake)

    # adv_loss = AdversarialLoss()
    # dm_loss = DomainClassificationLoss()
    # rec_loss = ReconstructionLoss()

    # loss_adv = adv_loss(fake_logits=output, real_logits=images)
    # loss_dm = dm_loss(logit=torch.squeeze(output_cls), target=labels)
    # loss_rec = rec_loss(real=images, reconstructed=output)
    # print(f"AdversarialLoss = {loss_adv}")
    # print(f"DomainClassificationLoss = {loss_dm}")
    # print(f"ReconstructionLoss = {loss_rec}")

    gen_loss = GeneratorLoss(discriminator=model.D)
    print(gen_loss(images, fake, fake, labels))
