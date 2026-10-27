"""CIFAR-10 split and loaders used in notebooks 05 and 06."""

from types import SimpleNamespace

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

NUM_TRAIN_IMAGES = 50000
NUM_VAL_IMAGES = 5000

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def split_indices(seed=42):
    """Return (val_indices, train_indices): first 5,000 of a seeded permutation."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(NUM_TRAIN_IMAGES, generator=generator).tolist()
    val_indices, train_indices = indices[:NUM_VAL_IMAGES], indices[NUM_VAL_IMAGES:]

    assert set(train_indices).isdisjoint(val_indices)

    return val_indices, train_indices


def get_cifar10(data_root, device, seed=42, batch_size=128, num_workers=0):
    """Build the 45k/5k/10k train/val/test split and loaders.

    The train loader is returned as a function so each training run can get
    a freshly seeded shuffle order.
    """
    data_root = str(data_root)
    pin_memory = torch.device(device).type == "cuda"

    train_data_augmented = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform,
    )
    train_data_eval = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=eval_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=eval_transform,
    )

    val_indices, train_indices = split_indices(seed)
    train_subset = Subset(train_data_augmented, train_indices)
    val_subset = Subset(train_data_eval, val_indices)

    def make_train_loader(seed=seed):
        return DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return SimpleNamespace(
        train_subset=train_subset,
        val_subset=val_subset,
        test_dataset=test_dataset,
        make_train_loader=make_train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=test_dataset.classes,
    )
