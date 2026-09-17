import os
from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Default directory for extracted datasets (relative to repository root)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "train"

def get_data_loaders(data_dir: str = str(DEFAULT_DATA_DIR), batch_size: int = 32, val_split: float = 0.2, test_split: float = 0.1, img_size: int = 224, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders.

    Args:
        data_dir: Root directory containing class subfolders.
        batch_size: Batch size for loaders.
        val_split: Fraction of data for validation.
        test_split: Fraction of data for test.
        img_size: Target image size (square).
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Training transforms with subtle data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Validation & Test transforms (deterministic)
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dir_path = Path(data_dir)
    # Check if pre-partitioned train/val subdirectories exist
    if (dir_path / "train").exists() and (dir_path / "val").exists():
        train_set = datasets.ImageFolder(root=str(dir_path / "train"), transform=train_transform)
        val_set = datasets.ImageFolder(root=str(dir_path / "val"), transform=eval_transform)
        test_path = dir_path / "test" if (dir_path / "test").exists() else dir_path / "val"
        test_set = datasets.ImageFolder(root=str(test_path), transform=eval_transform)
    else:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
        total_len = len(full_dataset)
        test_len = int(total_len * test_split)
        val_len = int(total_len * val_split)
        train_len = total_len - val_len - test_len
        train_set, val_set, test_set = random_split(full_dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader
