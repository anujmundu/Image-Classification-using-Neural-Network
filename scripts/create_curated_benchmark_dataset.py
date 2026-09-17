# scripts/create_curated_benchmark_dataset.py
"""Curate a balanced 50-category diverse subset from Caltech-256
and precisely partition it into:
- 70% Training
- 15% Validation
- 15% Testing
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]

# Set fixed seed for exact reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SOURCE_DIR = REPO_ROOT / "data" / "extracted" / "256_ObjectCategories" / "256_ObjectCategories"
TARGET_DIR = REPO_ROOT / "data" / "curated_benchmark_50"

# 50 Semantically Diverse Categories spanning Animals, Vehicles, Everyday Objects, Nature, Food, and Sports
SELECTED_50_CATEGORIES = [
    # Animals (14)
    "009.bear", "024.butterfly", "028.camel", "056.dog", "061.duck",
    "065.elephant-101", "080.frog", "084.giraffe", "087.gorilla", "105.horse",
    "113.hummingbird", "116.iguana", "129.leopard-101", "250.zebra",
    # Vehicles & Transportation (10)
    "014.blimp", "023.bulldozer", "030.canoe", "032.cartman", "098.harmonica",
    "101.helicopter-101", "145.motorbikes-101", "178.school-bus", "180.screwdriver", "195.speedboat",
    # Everyday Items & Electronics (10)
    "003.backpack", "012.binoculars", "027.calculator", "042.coffin", "047.computer-mouse",
    "089.goose", "091.grand-piano-101", "093.guitar-pick", "109.hot-tub", "125.laptop-101",
    # Nature, Food & Household (10)
    "015.bonsai-101", "025.cactus", "026.cake", "046.computer-keyboard", "054.diamond-ring",
    "086.golden-gate-bridge", "099.harpsichord", "108.hot-dog", "119.jesus-christ", "159.people",
    # Sports & Recreation (6)
    "004.baseball-bat", "005.baseball-glove", "006.basketball-hoop", "017.bowling-ball", "018.bowling-pin", "019.boxing-glove"
]

def curate_dataset(num_per_class: int = 100, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15):
    print("=" * 80)
    print("CURATING 50-CATEGORY BALANCED BENCHMARK DATASET")
    print(f"Source Directory: {SOURCE_DIR}")
    print(f"Target Directory: {TARGET_DIR}")
    print(f"Split Ratios: Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%")
    print("=" * 80)

    if not SOURCE_DIR.exists():
        print(f"Error: Source directory {SOURCE_DIR} does not exist!")
        sys.exit(1)

    # Clean target directory if already exists
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    train_dir = TARGET_DIR / "train"
    val_dir = TARGET_DIR / "val"
    test_dir = TARGET_DIR / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Discover and match categories
    available_dirs = {d.name: d for d in SOURCE_DIR.iterdir() if d.is_dir()}
    
    matched_categories = []
    for cat in SELECTED_50_CATEGORIES:
        if cat in available_dirs:
            matched_categories.append(cat)
        else:
            # Fallback prefix matching
            prefix = cat.split(".")[0]
            found = [d for d in available_dirs.keys() if d.startswith(f"{prefix}.")]
            if found:
                matched_categories.append(found[0])

    # If some categories were missing, top-up to exactly 50
    if len(matched_categories) < 50:
        for d_name in sorted(available_dirs.keys()):
            if d_name not in matched_categories and not d_name.startswith("257"):
                matched_categories.append(d_name)
                if len(matched_categories) == 50:
                    break

    matched_categories = matched_categories[:50]
    print(f"Selected {len(matched_categories)} categories successfully.\n")

    summary = {
        "num_classes": len(matched_categories),
        "train_count": 0,
        "val_count": 0,
        "test_count": 0,
        "total_images": 0,
        "classes": [],
        "per_class_stats": {}
    }

    clean_class_names = []

    for cat_folder in matched_categories:
        src_cat_dir = SOURCE_DIR / cat_folder
        all_imgs = sorted([p for p in src_cat_dir.glob("*.jpg") if p.is_file()])

        if len(all_imgs) < 30:
            print(f"⚠️ Warning: {cat_folder} has only {len(all_imgs)} images.")

        random.shuffle(all_imgs)

        # Cap at num_per_class if specified and available
        if num_per_class and len(all_imgs) > num_per_class:
            selected_imgs = all_imgs[:num_per_class]
        else:
            selected_imgs = all_imgs

        total_n = len(selected_imgs)
        n_train = int(total_n * train_ratio)
        n_val = int(total_n * val_ratio)
        n_test = total_n - n_train - n_val

        train_imgs = selected_imgs[:n_train]
        val_imgs = selected_imgs[n_train:n_train + n_val]
        test_imgs = selected_imgs[n_train + n_val:]

        # Clean folder name for pretty classification
        clean_name = cat_folder.split(".", 1)[-1] if "." in cat_folder else cat_folder
        clean_class_names.append(clean_name)

        # Create subdirectories
        cat_train = train_dir / clean_name
        cat_val = val_dir / clean_name
        cat_test = test_dir / clean_name

        cat_train.mkdir(parents=True, exist_ok=True)
        cat_val.mkdir(parents=True, exist_ok=True)
        cat_test.mkdir(parents=True, exist_ok=True)

        # Copy images
        for p in train_imgs:
            shutil.copy2(p, cat_train / p.name)
        for p in val_imgs:
            shutil.copy2(p, cat_val / p.name)
        for p in test_imgs:
            shutil.copy2(p, cat_test / p.name)

        summary["train_count"] += len(train_imgs)
        summary["val_count"] += len(val_imgs)
        summary["test_count"] += len(test_imgs)
        summary["total_images"] += total_n

        summary["per_class_stats"][clean_name] = {
            "source_folder": cat_folder,
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
            "total": total_n
        }

    summary["classes"] = clean_class_names

    # Save classes.json and metadata summary
    with open(TARGET_DIR / "classes.json", "w", encoding="utf-8") as f:
        json.dump(clean_class_names, f, indent=2)

    with open(TARGET_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 80)
    print("[SUCCESS] DATASET CURATION COMPLETE!")
    print(f"Total Categories: {summary['num_classes']}")
    print(f"Total Images:     {summary['total_images']}")
    print(f"  • Training (70%):   {summary['train_count']} images ({summary['train_count']/summary['num_classes']:.1f} per class)")
    print(f"  • Validation (15%): {summary['val_count']} images ({summary['val_count']/summary['num_classes']:.1f} per class)")
    print(f"  • Testing (15%):    {summary['test_count']} images ({summary['test_count']/summary['num_classes']:.1f} per class)")
    print(f"Target Output:    {TARGET_DIR}")
    print("-" * 80)

if __name__ == "__main__":
    curate_dataset(num_per_class=100)
