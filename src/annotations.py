# src/annotations.py
"""Utility for ingesting annotation files (COCO JSON, Pascal VOC XML, etc.).

- Places raw annotation files into `data/annotations/`.
- Validates file extensions.
- Moves processed files to `data/annotations/processed/`.
- Updates manifest JSONs to include annotation paths (optional, placeholder).
"""

import os
import shutil
from pathlib import Path
import json

DATA_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_DIR = DATA_ROOT / "data" / "annotations"
PROCESSED_ANN_DIR = ANNOTATIONS_DIR / "processed"

SUPPORTED_EXT = {".json", ".xml"}

def ingest_annotation(file_path: str) -> None:
    """Validate and move a single annotation file.
    Args:
        file_path: Path to the annotation file provided by the user.
    """
    src = Path(file_path)
    if not src.is_file():
        raise FileNotFoundError(f"Annotation file not found: {src}")
    if src.suffix.lower() not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported annotation format: {src.suffix}. Supported: {SUPPORTED_EXT}")
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    dest = ANNOTATIONS_DIR / src.name
    shutil.move(str(src), str(dest))
    print(f"[INFO] Moved {src.name} to {dest}")

def process_all() -> None:
    """Move every annotation file in `data/annotations/` to the processed sub‑folder.
    This can be called after the user has placed files manually.
    """
    PROCESSED_ANN_DIR.mkdir(parents=True, exist_ok=True)
    for ann_file in ANNOTATIONS_DIR.iterdir():
        if ann_file.is_file() and ann_file.suffix.lower() in SUPPORTED_EXT:
            dest = PROCESSED_ANN_DIR / ann_file.name
            shutil.move(str(ann_file), str(dest))
            print(f"[INFO] Processed {ann_file.name}")

def update_manifests() -> None:
    """Placeholder: Extend manifests to reference annotation files.
    Implementation depends on the annotation schema (e.g., COCO)."""
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest annotation files into the project.")
    parser.add_argument("files", nargs="*", help="Path(s) to annotation files to ingest")
    args = parser.parse_args()
    for f in args.files:
        ingest_annotation(f)
    process_all()
    update_manifests()
    print("[INFO] Annotation ingestion completed.")
