"""MedMNIST loading (28x28 grayscale, ToTensor only, so pixels are in [0, 1])."""

import os

import medmnist
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import DATA_DIR, DATASET_REGISTRY


def _dataset_class(dataset_name):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY)}")
    return getattr(medmnist, DATASET_REGISTRY[dataset_name][1])


def get_dataloaders(cfg):
    """(train_loader, val_loader, test_loader, n_classes) for cfg["dataset"]; train is shuffled."""
    dataset_cfg = cfg["dataset"]
    dataset_name = dataset_cfg.get("name", "pneumonia_mnist")
    dataset_class = _dataset_class(dataset_name)
    n_classes = len(medmnist.INFO[DATASET_REGISTRY[dataset_name][0]]["label"])

    root = dataset_cfg.get("data_root", str(DATA_DIR))
    os.makedirs(root, exist_ok=True)
    splits = {}
    for split in ("train", "val", "test"):
        splits[split] = dataset_class(split=split, download=dataset_cfg.get("download", True),
                                      root=root, transform=transforms.ToTensor())

    batch_size = dataset_cfg.get("batch_size", 32)
    num_workers = dataset_cfg.get("num_workers", 2)
    train_loader = DataLoader(splits["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(splits["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(splits["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, n_classes


def load_medmnist_dataset(dataset_name, split, data_root=None, download=True):
    """One MedMNIST split as a dataset object, in the official order."""
    root = str(DATA_DIR) if data_root is None else data_root
    os.makedirs(root, exist_ok=True)
    return _dataset_class(dataset_name)(split=split, download=download, root=root,
                                        transform=transforms.ToTensor())
