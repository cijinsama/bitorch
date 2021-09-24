import os

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .base import BasicDataset, Augmentation


class ImageNet(BasicDataset):
    name = "imagenet"
    num_classes = 1000
    shape = (1, 3, 224, 224)
    mean = (0.485, 0.456, 0.406)
    std_dev = (0.229, 0.224, 0.255)
    num_train_samples = 1281167
    num_val_samples = 50000

    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)

    def get_dataset(self, download: bool) -> Dataset:
        split = "train" if self.is_train else "val"
        directory = os.path.join(self.root_directory, split)
        if download and not os.path.isdir(directory):
            raise RuntimeError("ImageNet dataset must be downloaded and prepared manually.")
        return ImageFolder(directory, transform=self.get_transform())

    @classmethod
    def train_transform(cls, augmentation: Augmentation = Augmentation.DEFAULT) -> transforms.Compose:
        crop_scale = 0.08
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cls.get_normalize_transform(),
        ])

    @classmethod
    def test_transform(cls) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            cls.get_normalize_transform(),
        ])
