import os
from pathlib import Path

# Attempt to import torchvision; if unavailable, we'll skip CIFAR-10 download
try:
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
except ImportError:
    datasets = None
    transforms = None
import argparse
import sys

def download_cifar10(data_root: Path):
    if datasets is None:
        print("torchvision not available – skipping CIFAR-10 download. Ensure CIFAR-10 archives are present in the data directory.")
        return
    print("Downloading CIFAR-10...")
    # Use torchvision to download if available
    datasets.CIFAR10(root=data_root, train=True, download=True, transform=transforms.ToTensor())
    datasets.CIFAR10(root=data_root, train=False, download=True, transform=transforms.ToTensor())
    print("CIFAR-10 downloaded to", data_root)

def create_imagenet_placeholder(data_root: Path):
    # We cannot download full ImageNet here; create a placeholder for a subset.
    subset_path = data_root / "imagenet_subset"
    subset_path.mkdir(parents=True, exist_ok=True)
    # Optionally, you could download a small public subset (e.g., 100 classes) from a URL.
    # For now we just inform the user.
    print(f"Created placeholder for ImageNet subset at {subset_path}. Please place your subset images there (class subfolders).")

def create_custom_catalog_placeholder(data_root: Path):
    catalog_path = data_root / "custom_catalog"
    catalog_path.mkdir(parents=True, exist_ok=True)
    print(f"Created placeholder for custom e‑commerce catalog at {catalog_path}. Populate it with class‑wise image folders.")

def main():
    parser = argparse.ArgumentParser(description="Download datasets and create placeholders for ImageNet and custom catalog.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root directory for all datasets.")
    args = parser.parse_args()
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Attempt to download CIFAR-10 only if torchvision is present
    download_cifar10(root)
    create_imagenet_placeholder(root)
    create_custom_catalog_placeholder(root)

if __name__ == "__main__":
    main()
