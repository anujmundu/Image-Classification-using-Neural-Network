# src/preprocess.py
"""Data preprocessing for Image Classification project.

- Loads archives from `data/raw/` (if not yet extracted).
- Extracts archives into `data/extracted/`.
- Resizes images to a target size (default 224x224).
- Normalizes pixel values using ImageNet mean/std.
- Splits the dataset into train/validation sets (default 80/20).
- Saves processed images under `data/processed/{train,val}` and writes JSON manifests.
"""

import os
import json
import random
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

# Configuration – could be externalized to a config file
TARGET_SIZE = (224, 224)
TRAIN_RATIO = 0.8
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Resolve project root (two levels up from this file)
DATA_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = DATA_ROOT / "data" / "raw"
EXTRACTED_DIR = DATA_ROOT / "data" / "extracted"
PROCESSED_DIR = DATA_ROOT / "data" / "processed"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"

def extract_archives() -> None:
    """Extract any archive files found in RAW_DIR into a sub‑folder of EXTRACTED_DIR.
    Each archive creates its own folder named after the archive (without extension)."""
    archives = list(RAW_DIR.iterdir())
    for archive in tqdm(archives, desc="Extracting archives", unit="archive"):
        if not archive.is_file():
            continue
        dest = EXTRACTED_DIR / archive.stem
        dest.mkdir(parents=True, exist_ok=True)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(dest)
        elif archive.suffix in {".gz", ".tar"} or archive.name.endswith('.tar.gz'):
            # tarfile handles .tar, .gz, .tar.gz
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(dest)
        else:
            print(f"[WARN] Unsupported archive type: {archive}")

def _gather_image_paths(root: Path) -> List[Tuple[Path, str]]:
    """Recursively collect image file paths and infer class label from the immediate sub‑folder.
    Returns a list of (image_path, class_name)."""
    data = []
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_path in class_dir.rglob("*.[jp][pn]g"):
            data.append((img_path, class_name))
    return data

def _process_and_save(image_info: Tuple[Path, str], dst_root: Path) -> None:
    img_path, label = image_info
    # Ensure destination class folder exists
    class_dst = dst_root / label
    class_dst.mkdir(parents=True, exist_ok=True)
    # Load, resize, convert to RGB
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(TARGET_SIZE, Image.BILINEAR)
        # Save processed image (normalization will be applied later in the training pipeline)
        img.save(class_dst / img_path.name)

def split_and_process() -> None:
    """Split the dataset into train/val, process images, and write manifests."""
    all_images = _gather_image_paths(EXTRACTED_DIR)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * TRAIN_RATIO)
    train_set = all_images[:split_idx]
    val_set = all_images[split_idx:]

    # Process training images with progress bar
    for img_info in tqdm(train_set, desc="Processing train images", unit="img"):
        _process_and_save(img_info, TRAIN_DIR)
    # Process validation images with progress bar
    for img_info in tqdm(val_set, desc="Processing val images", unit="img"):
        _process_and_save(img_info, VAL_DIR)

    def _make_manifest(dataset: List[Tuple[Path, str]], base_dir: Path) -> List[dict]:
        manifest = []
        for img_path, label in dataset:
            rel_path = os.path.relpath(base_dir / label / img_path.name, DATA_ROOT)
            manifest.append({"image": rel_path.replace(os.sep, "/"), "label": label})
        return manifest

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "train_split.json", "w", encoding="utf-8") as f:
        json.dump(_make_manifest(train_set, TRAIN_DIR), f, indent=2)
    with open(PROCESSED_DIR / "val_split.json", "w", encoding="utf-8") as f:
        json.dump(_make_manifest(val_set, VAL_DIR), f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Data preprocessing for image classification")
    parser.add_argument("--target-size", type=int, nargs=2, default=TARGET_SIZE,
                        help="Target width and height for resizing images (default: 224 224)")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO,
                        help="Proportion of data to use for training (default: 0.8)")
    return parser.parse_args()


def main():
    args = parse_args()
    global TARGET_SIZE, TRAIN_RATIO
    TARGET_SIZE = tuple(args.target_size)
    TRAIN_RATIO = args.train_ratio
    print("[INFO] Extracting archives...")
    extract_archives()
    print("[INFO] Preprocessing images and creating split...")
    split_and_process()
    print("[INFO] Done. Manifests are stored in data/processed/.")

if __name__ == "__main__":
    main()
