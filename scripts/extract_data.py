# scripts/extract_data.py
"""Utility to extract dataset archives.

The repository now stores raw archives in ``data/raw`` and annotation archives in ``data/annotations``.
Running this script will unpack all archives into ``data/extracted`` (or ``data/annotations`` for annotation files).

Usage:
    python scripts/extract_data.py            # extract everything
    python scripts/extract_data.py --only caltech-101.zip cifar-10-python.tar.gz
"""

import argparse
import os
import tarfile
import zipfile
from pathlib import Path

def extract_tar(path: Path, dest: Path):
    with tarfile.open(path, 'r:*') as tar:
        tar.extractall(dest)

def extract_zip(path: Path, dest: Path):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(dest)

def main():
    parser = argparse.ArgumentParser(description="Extract dataset archives.")
    parser.add_argument(
        "--only",
        nargs="*",
        help="List of archive filenames to extract (default: all).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "data" / "raw"
    annotations_dir = repo_root / "data" / "annotations"
    extracted_dir = repo_root / "data" / "extracted"

    extracted_dir.mkdir(parents=True, exist_ok=True)

    # Determine which files to process
    if args.only:
        targets = [raw_dir / name for name in args.only]
    else:
        targets = list(raw_dir.iterdir())

    for archive in targets:
        if not archive.is_file():
            print(f"[WARN] Skipping non‑file: {archive}")
            continue
        print(f"[INFO] Extracting {archive.name} ...")
        try:
            if archive.suffix == ".zip":
                extract_zip(archive, extracted_dir)
            elif archive.suffix in {".tar", ".gz", ".tgz"} or archive.name.endswith('.tar.gz'):
                extract_tar(archive, extracted_dir)
            else:
                print(f"[WARN] Unknown archive type: {archive.name}")
                continue
        except Exception as e:
            print(f"[ERROR] Failed to extract {archive.name}: {e}")
            continue

    # Extract annotation archives (if any) into the annotations folder
    for ann_archive in annotations_dir.iterdir():
        if ann_archive.is_file():
            print(f"[INFO] Extracting annotation {ann_archive.name} ...")
            try:
                if ann_archive.suffix == ".zip":
                    extract_zip(ann_archive, annotations_dir)
                elif ann_archive.name.endswith('.tar') or ann_archive.name.endswith('.tar.gz'):
                    extract_tar(ann_archive, annotations_dir)
                else:
                    print(f"[WARN] Unknown annotation archive: {ann_archive.name}")
            except Exception as e:
                print(f"[ERROR] Failed to extract {ann_archive.name}: {e}")

if __name__ == "__main__":
    main()
