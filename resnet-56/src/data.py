"""CIFAR-10 splits, preprocessing, and loaders."""

from copy import copy
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


class SubtractMeanImage:
    def __init__(self, mean_image):
        self.mean_image = mean_image

    def __call__(self, image):
        return image - self.mean_image


def split_indices(seed=42):
    """Use the first 5,000 indices of a seeded permutation for validation."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(50_000, generator=generator).tolist()
    return indices[:5_000], indices[5_000:]


def get_cifar10(data_root, device, seed=42, batch_size=128, num_workers=0):
    """Return the training loader factory, validation loader, and training mean."""
    val_indices, train_indices = split_indices(seed)
    raw_data = CIFAR10(root=data_root, train=True, download=True)
    mean_hwc = raw_data.data[train_indices].mean(axis=0, dtype=np.float64) / 255.0
    mean_image = torch.from_numpy(mean_hwc).permute(2, 0, 1).float()

    center_image = transforms.Compose([transforms.ToTensor(), SubtractMeanImage(mean_image)])
    train_data = copy(raw_data)
    train_data.transform = transforms.Compose([
        center_image,
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    raw_data.transform = center_image
    train_subset = Subset(train_data, train_indices)
    val_subset = Subset(raw_data, val_indices)

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.device(device).type == "cuda",
    }

    def make_train_loader():
        return DataLoader(
            train_subset,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            **loader_options,
        )

    return SimpleNamespace(
        make_train_loader=make_train_loader,
        val_loader=DataLoader(val_subset, **loader_options),
        mean_image=mean_image,
        class_names=raw_data.classes,
        seed=seed,
    )


def get_test_loader(data_root, mean_image, device, batch_size=128, num_workers=0):
    """Load the held-out test set with the mean image saved during training."""
    test_data = CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), SubtractMeanImage(mean_image)]),
    )
    return DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.device(device).type == "cuda",
    )
