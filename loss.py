import torch
import torch.nn as nn

from model import Discriminator, Generator


class AdversarialLoss:
    """
    To make the generated images indistinguishable from real images
    """

    def __init__(self) -> None:

        super(AdversarialLoss, self).__init__()

    def __call__(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """
        :params torch.Tensor real: real image
        :params torch.Tensor fake: image after generator with label condition
        :rtype: torch.Tensor
        :returns: Adversarial loss of real and fake images
        """
        return torch.log(torch.mean(real)) + torch.log(1 - torch.mean(fake))