"""
Kaggle dataset downloader and staging utility.

Downloads datasets via kagglehub and copies them into the exact ./data/
subdirectory paths expected by KNOWN_RAW_SOURCES in incremental.py.

Usage: python main.py download
       python -m src.training.datasets
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

# Maps Kaggle dataset ID -> target path under DATA_DIR.
# Paths match the source_roots used in KNOWN_RAW_SOURCES exactly.
KAGGLE_DATASETS: List[Dict] = [
    {
        "id": "rkuo2000/uecfood256",
        "name": "uec_food_256",
        "target_subdir": "UECFOOD256",
    },
    {
        "id": "moltean/fruits",
        "name": "fruits_360",
        "target_subdir": "Fruits-360 dataset",
    },
    {
        "id": "kritikseth/fruit-and-vegetable-image-recognition",
        "name": "fruit_and_vegetable_image_recognition",
        "target_subdir": "Food Recognition 2022",
    },
    {
        "id": "zhaoyj688/vegfru",
        "name": "vegfru",
        "target_subdir": "vegfru",
    },
    {
        "id": "harishkumardatalab/food-image-classification-dataset",
        "name": "food_image_classification_dataset",
        "target_subdir": "FoodImageClassifactonDataset",
    },
    {
        "id": "iamsouravbanerjee/indian-food-images-dataset",
        "name": "indian_food_images_dataset",
        "target_subdir": "Indian Food Images Dataset",
    },
    {
        "id": "rizkyyk/dataset-food-classification",
        "name": "indonesian_food_dataset",
        "target_subdir": "Indonesian Food Dataset",
    },
    # Previously missing from download script — now included
    {
        "id": "utkarshsaxenadn/fast-food-classification-v2",
        "name": "fast_food_classification_dataset",
        "target_subdir": "fast food classifaction",
    },
]


def _copy_tree(src: Path, dst: Path) -> int:
    """
    Copy src into dst, preserving the src directory name as the last component.
    Returns the number of files copied.
    """
    target = dst / src.name
    if target.exists():
        print(f"  Already present: {target} — skipping copy")
        return 0

    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in src.rglob("*"):
        if item.is_file():
            rel = item.relative_to(src)
            dest_file = target / rel
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest_file)
            copied += 1

    return copied


def download_and_stage(data_dir: Optional[Path] = None, dry_run: bool = False) -> None:
    """
    Download all configured Kaggle datasets and copy them to the paths
    that KNOWN_RAW_SOURCES expects under data/.

    Requires:
        pip install kagglehub
        ~/.kaggle/kaggle.json  (your Kaggle API credentials)

    Args:
        data_dir: Override destination (defaults to data/ in project root).
        dry_run:  Print what would be done without downloading or copying.
    """
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Run: pip install kagglehub")
        return

    dest_root = data_dir or DATA_DIR
    dest_root.mkdir(parents=True, exist_ok=True)

    failed: List[Dict] = []

    for spec in KAGGLE_DATASETS:
        dataset_id = spec["id"]
        name = spec["name"]
        target_dir = dest_root / spec["target_subdir"]

        if target_dir.exists():
            print(f"[SKIP] {name}: target already exists at {target_dir}")
            continue

        if dry_run:
            print(f"[DRY RUN] Would download {dataset_id} -> {target_dir}")
            continue

        print(f"\nDownloading {name} ({dataset_id})...")
        try:
            cache_path = Path(kagglehub.dataset_download(dataset_id))
            print(f"  Downloaded to cache: {cache_path}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failed.append({"name": name, "id": dataset_id, "error": str(exc)})
            continue

        print(f"  Copying to {target_dir}...")
        target_dir.mkdir(parents=True, exist_ok=True)
        file_count = _copy_tree(cache_path, target_dir)
        if file_count == 0:
            # kagglehub may return a parent directory; try its children
            for child in cache_path.iterdir():
                if child.is_dir():
                    file_count += _copy_tree(child, target_dir)

        print(f"  Staged {file_count} files into {target_dir}")

    if failed:
        print("\nSome downloads failed:")
        for item in failed:
            print(f"  - {item['name']} ({item['id']}): {item['error']}")
    else:
        print("\nAll datasets downloaded and staged successfully.")


def main() -> None:
    print("NutriVision — Kaggle Dataset Download & Staging")
    print("=" * 60)
    print(f"Destination: {DATA_DIR}")
    print(f"Datasets: {len(KAGGLE_DATASETS)}")
    for spec in KAGGLE_DATASETS:
        status = "present" if (DATA_DIR / spec["target_subdir"]).exists() else "missing"
        print(f"  [{status:7s}] {spec['name']} ({spec['id']})")
    print()
    download_and_stage()


if __name__ == "__main__":
    main()
