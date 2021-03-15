from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from config import cfg

# 8 - blackhair
# 9 - blondhair
# 11 - brownhair
# 20 - male
# 22 - mustache
# 24 - nobeard
# 31 - smiling
# 39 - young

transforms = Compose(
    [
        Resize((cfg.image.height, cfg.image.width)),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

celeba = CelebA(
    cfg.dataloader.celebA,
    target_type=cfg.dataloader.target_type,
    transform=transforms,
    download=cfg.dataloader.download,
)
celeba_dataloader = DataLoader(
    celeba, batch_size=cfg.training.batch_size, shuffle=cfg.training.shuffle
)
