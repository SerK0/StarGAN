import warnings
from typing import Tuple

import torch
from torch import nn

from config import cfg
from discriminator import Discriminator
from generator import Generator
from loss import DiscriminatorLoss, GeneratorLoss
from utils import permute_labels

warnings.filterwarnings("ignore")


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
        self.generator_loss = GeneratorLoss(self.D)
        self.discriminator_loss = DiscriminatorLoss()
        self.optimizer_generator = torch.optim.Adam(
            self.G.parameters(), lr=cfg.training.G.lr, betas=cfg.training.G.betas
        )
        self.optimizer_dicriminator = torch.optim.Adam(
            self.D.parameters(), lr=cfg.training.D.lr, betas=cfg.training.D.betas
        )

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

    def trainG(
        self, real_image: torch.Tensor, labels_dataset: torch.Tensor
    ) -> torch.Tensor:
        """
        :param torch.Tensor real_image: real image from dataset
        :param torch.Tensor labels_dataset: real domain labels to image from dataset
        :rtype: torch.Tensor
        :returns: Generator loss
        """
        self.G.train()
        self.D.eval()

        labels_target = permute_labels(labels_dataset)
        fake_image = self.G(self.concat_image_label(real_image, labels_target))
        out_src, out_cls = self.D(fake_image)
        reconstructed_image = self.G(
            self.concat_image_label(fake_image, labels_dataset)
        )

        loss = self.generator_loss(
            real=real_image,
            fake=fake_image,
            reconstructed=reconstructed_image,
            labels_target=labels_target,
        )

        self.optimizer_generator.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), cfg.training.G.clipping)
        self.optimizer_generator.step()

        return loss.item()

    def trainD(
        self, real_image: torch.Tensor, labels_dataset: torch.Tensor
    ) -> torch.Tensor:
        """
        :param torch.Tensor real_image: real image from dataset
        :param torch.Tensor labels_dataset: real domain labels to image from dataset
        :rtype: torch.Tensor
        :returns: Discriminator loss
        """
        self.G.eval()
        self.D.train()

        labels_target = permute_labels(labels_dataset)
        fake_image = self.G(self.concat_image_label(real_image, labels_target))
        out_src_real, out_cls_real = self.D(real_image)
        out_src_fake, out_cls_fake = self.D(fake_image)

        loss = self.discriminator_loss(
            out_src_real, out_src_fake, out_cls_real, labels_dataset
        )

        #####TODO Compute loss for gradient penalty.
        self.optimizer_dicriminator.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), cfg.training.D.clipping)
        loss.backward()
        self.optimizer_dicriminator.step()

        return loss.item()

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

    batch_size, channels, height, width = (4, 3, 128, 128)
    n_domain = 5
    real_image = torch.rand(batch_size, channels, height, width)
    labels_dataset = torch.rand(batch_size, n_domain)
    stargan = StarGan(
        height=height, width=width, n_domain=n_domain, in_channels=channels
    )

    stargan.train()
    # loss_gen = stargan.trainG(real_image, labels_dataset)
    # print(loss_gen)
    loss_dis = stargan.trainD(real_image, labels_dataset)
    print(loss_dis)
