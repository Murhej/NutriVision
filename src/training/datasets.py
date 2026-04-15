"""
Kaggle dataset downloader and staging utility.

Downloads datasets via kagglehub and copies them into the exact ./data/
subdirectory paths expected by KNOWN_RAW_SOURCES in incremental.py.

Usage: python main.py download # default: link data/ into kagglehub cache
       python main.py download --full-copy # standalone copy into data/
       python -m src.training.datasets
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
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


def _count_files_under(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file())


def _try_directory_link(link_path: Path, target_dir: Path) -> bool:
    """
    Make link_path point at target_dir (must be a directory).
    Windows: directory junction (no extra privileges). POSIX: symlink.
    """
    if link_path.exists() or not target_dir.is_dir():
        return False
    link_path.parent.mkdir(parents=True, exist_ok=True)
    src_abs = target_dir.resolve()
    try:
        if os.name == "nt":
            proc = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link_path), str(src_abs)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return proc.returncode == 0 and link_path.is_dir()
        link_path.symlink_to(src_abs, target_is_directory=True)
        return link_path.is_dir()
    except OSError:
        return False


def _copy_tree(src: Path, dst: Path, *, prefer_hardlink: bool = False) -> int:
    """
    Copy src into dst, preserving the src directory name as the last component.
    If prefer_hardlink is True, try os.link per file (same volume = instant),
    then fall back to copy2.
    Returns the number of files written/linked.
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
            if prefer_hardlink:
                try:
                    os.link(item, dest_file)
                except OSError:
                    shutil.copy2(item, dest_file)
            else:
                shutil.copy2(item, dest_file)
            copied += 1

    return copied


def _stage_kaggle_extract(cache_path: Path, target_dir: Path, *, fast_stage: bool) -> int:
    """
    Mirror legacy layout under target_dir (same as historical copy-only behavior).
    fast_stage: try a single directory junction/symlink per extracted folder first.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    inner = target_dir / cache_path.name
    if inner.exists():
        return _count_files_under(inner)

    if fast_stage and _try_directory_link(inner, cache_path):
        n = _count_files_under(cache_path)
        print(f"  Linked {inner.name} -> kagglehub cache ({n} files; no duplicate copy).")
        return n

    file_count = _copy_tree(cache_path, target_dir, prefer_hardlink=fast_stage)
    if file_count == 0:
        for child in sorted(cache_path.iterdir()):
            if not child.is_dir():
                continue
            inner_c = target_dir / child.name
            if inner_c.exists():
                file_count += _count_files_under(inner_c)
                continue
            if fast_stage and _try_directory_link(inner_c, child):
                n = _count_files_under(child)
                print(f"  Linked {inner_c.name} -> kagglehub cache ({n} files; no duplicate copy).")
                file_count += n
                continue
            file_count += _copy_tree(child, target_dir, prefer_hardlink=fast_stage)
    return file_count


def download_and_stage(
    data_dir: Optional[Path] = None,
    dry_run: bool = False,
    fast_stage: bool = True,
) -> None:
    """
    Download all configured Kaggle datasets and copy them to the paths
    that KNOWN_RAW_SOURCES expects under data/.

    Requires:
        pip install kagglehub
        ~/.kaggle/kaggle.json  (your Kaggle API credentials)

    Args:
        data_dir: Override destination (defaults to data/ in project root).
        dry_run:  Print what would be done without downloading or copying.
        fast_stage: If True (default), prefer directory junction (Windows) / symlink (POSIX)
            into the kagglehub cache, or per-file hardlinks on the same volume. Staged paths
            under data/ then depend on that cache — do not delete the kagglehub cache while using them.
            If False, perform a full file copy into data/ (standalone; slower).
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

        if fast_stage:
            print(f"  Staging into {target_dir} (fast: link when possible)...")
        else:
            print(f"  Copying to {target_dir}...")
        target_dir.mkdir(parents=True, exist_ok=True)
        file_count = _stage_kaggle_extract(cache_path, target_dir, fast_stage=fast_stage)

        print(f"  Staged {file_count} files into {target_dir}")

    if failed:
        print("\nSome downloads failed:")
        for item in failed:
            print(f"  - {item['name']} ({item['id']}): {item['error']}")
    else:
        print("\nAll datasets downloaded and staged successfully.")


def main(full_copy: bool = False) -> None:
    env_full = os.environ.get("NUTRIVISION_DOWNLOAD_FULL_COPY", "").strip().lower() in ("1", "true", "yes")
    env_disable_link = os.environ.get("NUTRIVISION_DOWNLOAD_FAST", "").strip().lower() in ("0", "false", "no")
    fast_stage = not (full_copy or env_full or env_disable_link)

    print("NutriVision — Kaggle Dataset Download & Staging")
    print("=" * 60)
    print(f"Destination: {DATA_DIR}")
    if fast_stage:
        print("Mode: link into kagglehub cache (junction/symlink/hardlink) — data/ is bound to that cache; keep it.")
    else:
        print("Mode: full copy into data/ (standalone; not tied to cache location).")
    print(f"Datasets: {len(KAGGLE_DATASETS)}")
    for spec in KAGGLE_DATASETS:
        status = "present" if (DATA_DIR / spec["target_subdir"]).exists() else "missing"
        print(f"  [{status:7s}] {spec['name']} ({spec['id']})")
    print()
    download_and_stage(fast_stage=fast_stage)


if __name__ == "__main__":
    main(full_copy="--full-copy" in sys.argv)
