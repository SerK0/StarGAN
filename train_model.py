import random

import numpy as np
import torch
from imageio import imsave

from config import cfg
from dataloader import celeba, celeba_dataloader
from stargan import StarGan
from utils import permute_labels
from torchvision.utils import save_image

np.random.seed(2020)
random.seed(2020)
torch.manual_seed(2020)


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def train():

    stargan = StarGan(
        height=cfg.image.height,
        width=cfg.image.width,
        n_domain=len(cfg.image.labels),
        in_channels=cfg.image.channels,
        device=cfg.device,
    )

    stargan.train()

    celeba_iter = iter(celeba_dataloader)
    real_image_test, attr_test = next(celeba_iter)
    labels_dataset_test = attr_test[:, cfg.image.labels].float()
    labels_target_test = permute_labels(labels_dataset_test)

    for epoch in range(cfg.training.n_epoch):

        for index, (real_image, attr) in enumerate(celeba_dataloader):
            real_image = real_image.float()
            labels_dataset = attr[:, cfg.image.labels].float()
            labels_target = permute_labels(labels_dataset)

            d_loss = stargan.trainD(real_image, labels_dataset, labels_target)

            if (index + 1) % cfg.training.D.n_updates == 0:
                g_loss = stargan.trainG(real_image, labels_dataset, labels_target)

            if (index + 1) % cfg.training.log_step == 0:
                print(f"Discriminator loss:{d_loss}|  |Generator loss:{g_loss}")

            if (index + 1) % cfg.training.sample_step == 0:
                real_image_test_list = [real_image_test]

                for label_target_test in labels_target_test:
                    fake_images = stargan.generate(
                        real_image_test,
                        label_target_test.repeat(real_image_test.size(0), 1),
                    )
                    real_image_test_list.append(fake_images)
                real_image_test_concat = torch.cat(real_image_test_list, dim=3)

                save_image(
                    denorm(real_image_test_concat.data.cpu()),
                    cfg.results.generated.format(index + 1),
                    nrow=1,
                    padding=0,
                )

            if (index + 1) % cfg.training.save_step == 0:
                torch.save(stargan, cfg.results.model)
        break


if __name__ == "__main__":
    train()
