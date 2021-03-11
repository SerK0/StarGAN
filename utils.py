import torch
from torch import nn


def compute_gradient_penalty(critic, real_samples, fake_samples):
    # return gradient_penalty
    pass


def permute_labels(labels_dataset: torch.Tensor) -> torch.Tensor:
    """
    :param torch.Tensor labels_dataset: original labels from dataset
    :rtype: torch.Tensor
    :returns: randomly generated labels
    """
    rand_idx = torch.randperm(labels_dataset.size(0))
    label_target = labels_dataset[rand_idx]

    return label_target