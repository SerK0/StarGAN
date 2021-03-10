import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits

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
        return torch.mean(real) + torch.mean(fake)


class DomainClassificationLoss:
    """
    For a given input image x and a target domain label c, our goal is to translate x into
    an output image y, which is properly classified to the target domain c
    """

    def __init__(self) -> None:

        super(DomainClassificationLoss, self).__init__()

    def __call__(self, logit, target: torch.Tensor) -> torch.Tensor:
        """
        :params torch.Tensor logit: logits from cls head of discriminator
        :params torch.Tensor target: true labels
        :rtype: torch.Tensor
        :returns: Domain classification loss of image
        """
        return binary_cross_entropy_with_logits(
            logit, target, size_average=False
        ) / logit.size(0)
