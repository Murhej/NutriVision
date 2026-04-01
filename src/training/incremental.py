"""
Incremental fine-tuning on top of the current best Food-101 checkpoint.

Handles dataset discovery, class merging, Food-101 replay (anti-forgetting),
and checkpoint promotion with a degradation guard.

Entry point: python main.py incremental
             python -m src.training.incremental

Expected custom dataset layout:
    data/custom_incremental/
        train/
            apple/
            banana/
        val/   # optional
        test/  # optional
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.datasets import Food101, ImageFolder

from src.core.device import get_device, log_environment_info, set_seed
from src.core.model import build_model, classifier_head_keys
from src.core.transforms import get_transforms
from src.training.config import Config as TrainConfig
from src.training.config import IncrementalConfig
from src.training.trainer import train_model


# ---------------------------------------------------------------------------
# Custom dataset helpers
# ---------------------------------------------------------------------------

class LabelRemapDataset(Dataset):
    """Re-index labels from a dataset's local numbering into the merged class space."""

    def __init__(self, dataset: Dataset, index_to_name: Sequence[str], class_to_index: Dict[str, int]):
        self.dataset = dataset
        self.index_to_name = list(index_to_name)
        self.class_to_index = class_to_index

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        class_name = self.index_to_name[int(label)]
        return image, self.class_to_index[class_name]


class SamplesDataset(Dataset):
    """Dataset from an explicit list of (path, label) pairs — used for COCO-style sources."""

    def __init__(self, samples: Sequence[Tuple[str, int]], class_names: Sequence[str], transform=None):
        self.samples = list(samples)
        self.classes = list(class_names)
        self.targets = [label for _, label in self.samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Known dataset registry
# ---------------------------------------------------------------------------

KNOWN_RAW_SOURCES = [
    {
        "name": "uec_food_256",
        "source_roots": [
            "./data/UECFOOD256",
            "./data/UEC-Food256/UECFOOD256",
            "./data/uecfood256/UECFOOD256",
        ],
        "target_root": "./data/uec_food_256_incremental",
        "train_subdir": ".",
    },
    {
        "name": "fruits_360",
        "source_roots": ["./data/Fruits-360 dataset/fruits-360_100x100/fruits-360"],
        "target_root": "./data/fruits_360_incremental",
        "train_subdir": "Training",
        "fallback_train_subdirs": ["Test"],
        "test_subdir": "Test",
    },
    {
        "name": "fruit_and_vegetable_image_recognition",
        "source_roots": ["./data/Food Recognition 2022/raw_data"],
        "target_root": "./data/fruitbot_expanded_incremental",
        "loader": "coco_flat",
        "train_subdir": "public_training_set_release_2.0",
        "val_subdir": "public_validation_set_2.0",
        "test_subdir": "public_test_release_2.0",
    },
    {
        "name": "vegfru",
        "source_roots": [
            "./data/vegfru/fru92_images",
            "./data/vegfru/veg200_images",
        ],
        "target_root": "./data/fruitbot_expanded_incremental",
        "train_subdir": ".",
    },
    {
        "name": "food_image_classification_dataset",
        "source_roots": ["./data/FoodImageClassifactonDataset/Food Classification dataset"],
        "target_root": "./data/custom_incremental",
        "train_subdir": ".",
    },
    {
        "name": "indian_food_images_dataset",
        "source_roots": ["./data/Indian Food Images Dataset/Indian Food Images"],
        "target_root": "./data/custom_incremental",
        "train_subdir": ".",
    },
    {
        "name": "indonesian_food_dataset",
        "source_roots": ["./data/Indonesian Food Dataset/dataset_gambar"],
        "target_root": "./data/custom_incremental",
        "train_subdir": "train",
        "val_subdir": "valid",
        "test_subdir": "test",
    },
    {
        "name": "fast_food_classification_dataset",
        "source_roots": ["./data/fast food classifaction/Fast Food Classification V2"],
        "target_root": "./data/custom_incremental",
        "train_subdir": "Train",
        "val_subdir": "Valid",
        "test_subdir": "Test",
    },
]


# ---------------------------------------------------------------------------
# Label normalisation and canonicalisation
# ---------------------------------------------------------------------------

FRUITS_360_EXACT_LABEL_ALIASES = {
    "dangshan_pear": "pear", "bergamot_pear": "pear", "crown_pear": "pear",
    "hami_melon": "melon", "melon_piel_de_sapo": "melon",
    "cherry_tomato": "tomato", "red_currant": "currant", "black_currant": "currant",
}
FRUITS_360_PREFIX_ALIASES = {
    "apple": "apple", "pear": "pear", "tomato": "tomato", "pepper": "pepper",
    "potato": "potato", "onion": "onion", "garlic": "garlic", "cucumber": "cucumber",
    "banana": "banana", "orange": "orange", "grape": "grape", "mango": "mango",
    "peach": "peach", "nectarine": "nectarine", "plum": "plum", "melon": "melon",
    "kiwi": "kiwi", "lemon": "lemon", "lime": "lime", "apricot": "apricot",
    "carrot": "carrot", "cabbage": "cabbage", "cauliflower": "cauliflower",
    "papaya": "papaya", "pineapple": "pineapple", "pomegranate": "pomegranate",
    "radish": "radish", "strawberry": "strawberry", "watermelon": "watermelon",
}
FOOD_RECOGNITION_ALLOWED_TOKENS = {
    "apple", "apricot", "asparagus", "avocado", "banana", "beans", "beef", "beetroot",
    "blueberries", "bread", "broccoli", "burger", "cabbage", "cake", "carrot", "cashew",
    "cauliflower", "cheese", "cherry", "chicken", "chickpeas", "chips", "chocolate",
    "cod", "couscous", "crackers", "cucumber", "curry", "egg", "eggplant", "fajita",
    "fish", "french", "fruit", "garlic", "gnocchi", "grape", "ham", "hamburger",
    "halloumi", "hummus", "ice", "kiwi", "lamb", "lasagne", "leek", "lentils", "lettuce",
    "mandarine", "mango", "meat", "melon", "mozzarella", "mushroom", "mushrooms",
    "noodles", "nuts", "oats", "olive", "olives", "omelette", "onion", "orange",
    "pancakes", "pasta", "peanut", "pear", "peas", "penne", "pepper", "pizza", "pork",
    "potato", "potatoes", "pretzel", "pumpkin", "quiche", "radish", "ratatouille",
    "ravioli", "rice", "salad", "salami", "salmon", "sandwich", "sausage", "shrimp",
    "soup", "spinach", "spaghetti", "spring", "steak", "strawberries", "sushi",
    "taboule", "tart", "tomato", "tuna", "turkey", "vegetable", "vegetables", "veggie",
    "watermelon", "wrap", "zucchini",
}
FOOD_RECOGNITION_BLOCKED_TOKENS = {
    "alcohol", "aperitif", "beer", "bouillon", "butter", "caffeine", "cocktail",
    "coffee", "cola", "concentrate", "dressing", "drink", "espresso", "herbal",
    "ice_cubes", "juice", "ketchup", "lemonade", "liquor", "mayonnaise", "mineral",
    "oil", "powder", "prosecco", "ristretto", "salt", "sauce", "soda", "syrup",
    "tea", "vinegar", "water", "wine",
}

DISCOVERY_SKIP_DIR_NAMES = {
    "food_101", "custom_incremental", "fruitbot_expanded_incremental",
    "fruits_360_incremental", "uec_food_256_incremental",
    "__pycache__", "images", "meta", "hub", "raw_data", "chunks",
    "chunks_index", "tiles_index", "images_meta", "boxes", "areas",
    "masks", "iscrowds", "super_categories", "categories",
}
TRAIN_DIR_ALIASES = {"train", "training"}
VAL_DIR_ALIASES = {"val", "valid", "validation"}
TEST_DIR_ALIASES = {"test", "testing"}


def normalize_label(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def canonicalize_source_label(source_name: str, raw_label: str) -> str:
    label = normalize_label(raw_label)
    if source_name == "fruits_360":
        label = re.sub(r"_\d+$", "", label)
        label = FRUITS_360_EXACT_LABEL_ALIASES.get(label, label)
        for prefix, canonical in FRUITS_360_PREFIX_ALIASES.items():
            if label == prefix or label.startswith(f"{prefix}_"):
                return canonical
    return label


def should_keep_source_label(source_name: str, label: str) -> bool:
    if source_name != "fruit_and_vegetable_image_recognition":
        return True
    tokens = set(label.split("_"))
    if tokens & FOOD_RECOGNITION_BLOCKED_TOKENS:
        return False
    return bool(tokens & FOOD_RECOGNITION_ALLOWED_TOKENS)


# ---------------------------------------------------------------------------
# File / path helpers
# ---------------------------------------------------------------------------

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def find_existing_source_root(spec: Dict) -> Optional[Path]:
    for root_text in spec["source_roots"]:
        root = Path(root_text)
        if root.exists():
            return root
    return None


def resolve_split_root(source_root: Path, split_name: Optional[str]) -> Optional[Path]:
    if split_name is None:
        return None
    if split_name == ".":
        return source_root
    split_root = source_root / split_name
    return split_root if split_root.exists() else None


def resolve_split_root_with_fallbacks(
    source_root: Path,
    split_name: Optional[str],
    fallback_split_names: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    split_root = resolve_split_root(source_root, split_name)
    if split_root is not None:
        return split_root
    for fallback in fallback_split_names or []:
        split_root = resolve_split_root(source_root, fallback)
        if split_root is not None:
            return split_root
    return None


def looks_like_imagefolder_root(root: Path, min_class_dirs: int) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    valid = 0
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        if any(p.is_file() and is_image_file(p) for p in class_dir.iterdir()):
            valid += 1
            if valid >= min_class_dirs:
                return True
    return False


def link_or_copy_file(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


# ---------------------------------------------------------------------------
# Auto-discovery of extra datasets
# ---------------------------------------------------------------------------

def discover_extra_data_roots(
    config: IncrementalConfig,
    excluded_source_roots: Sequence[Path],
) -> List[Dict]:
    if not config.auto_discover_extra_data_dirs:
        return []

    data_root = Path(config.data_dir).resolve()
    excluded_resolved = [r.resolve() for r in excluded_source_roots if r.exists()]
    discovered: List[Dict] = []
    seen: set[str] = set()

    for current_root, dir_names, _ in os.walk(data_root, topdown=True):
        current_path = Path(current_root)
        depth = len(current_path.relative_to(data_root).parts)
        if depth > config.discovery_max_depth:
            dir_names[:] = []
            continue

        if normalize_label(current_path.name) in DISCOVERY_SKIP_DIR_NAMES:
            dir_names[:] = []
            continue

        if any(is_under_root(current_path, excl) for excl in excluded_resolved):
            dir_names[:] = []
            continue

        child_map = {normalize_label(c.name): c for c in current_path.iterdir() if c.is_dir()}
        train_root = next((v for k, v in child_map.items() if k in TRAIN_DIR_ALIASES), None)
        val_root = next((v for k, v in child_map.items() if k in VAL_DIR_ALIASES), None)
        test_root = next((v for k, v in child_map.items() if k in TEST_DIR_ALIASES), None)

        if train_root and looks_like_imagefolder_root(train_root, config.discovery_min_class_dirs):
            key = str(current_path.resolve())
            if key not in seen:
                discovered.append({
                    "name": normalize_label(current_path.name),
                    "source_root": current_path,
                    "train_root": train_root,
                    "val_root": val_root,
                    "test_root": test_root,
                })
                seen.add(key)
            dir_names[:] = []
            continue

        parent_name = normalize_label(current_path.parent.name) if current_path.parent != current_path else ""
        if (
            parent_name not in TRAIN_DIR_ALIASES | VAL_DIR_ALIASES | TEST_DIR_ALIASES
            and looks_like_imagefolder_root(current_path, config.discovery_min_class_dirs)
        ):
            key = str(current_path.resolve())
            if key not in seen:
                discovered.append({
                    "name": normalize_label(current_path.name),
                    "source_root": current_path,
                    "train_root": current_path,
                    "val_root": None,
                    "test_root": None,
                })
                seen.add(key)
            dir_names[:] = []

    return discovered


# ---------------------------------------------------------------------------
# Dataset split helpers
# ---------------------------------------------------------------------------

def choose_subset_indices(total_size: int, requested_size: int) -> np.ndarray:
    if requested_size <= 0 or requested_size >= total_size:
        return np.arange(total_size)
    return np.random.choice(total_size, requested_size, replace=False)


def split_indices_by_class(
    targets: Sequence[int],
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    class_to_idx: Dict[int, List[int]] = {}
    for idx, tgt in enumerate(targets):
        class_to_idx.setdefault(int(tgt), []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for class_indices in class_to_idx.values():
        shuffled = np.array(class_indices, dtype=np.int64)
        np.random.shuffle(shuffled)
        total = len(shuffled)
        max_holdout = max(0, total - 1)

        val_count = min(int(round(total * val_ratio)), max_holdout)
        rem = max_holdout - val_count
        test_count = min(int(round(total * test_ratio)), rem)

        if val_ratio > 0 and val_count == 0 and total >= 3:
            val_count = 1
            rem = max(0, total - 1 - val_count)
            test_count = min(test_count, rem)
        if test_ratio > 0 and test_count == 0 and total - val_count >= 2 and rem > 0:
            test_count = 1

        val_idx.extend(shuffled[:val_count].tolist())
        test_idx.extend(shuffled[val_count:val_count + test_count].tolist())
        train_idx.extend(shuffled[val_count + test_count:].tolist())

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def load_extra_dataset_from_roots(
    source_name: str,
    source_root: Path,
    train_root: Path,
    val_root: Optional[Path],
    test_root: Optional[Path],
    config: IncrementalConfig,
    train_transform,
    test_transform,
) -> Dict:
    train_aug = ImageFolder(str(train_root), transform=train_transform)
    train_eval = ImageFolder(str(train_root), transform=test_transform)
    extra_class_names = [canonicalize_source_label(source_name, n) for n in train_aug.classes]

    test_indices: List[int] = []
    val_indices: List[int] = []

    if val_root is not None and val_root.exists():
        val_dataset = ImageFolder(str(val_root), transform=test_transform)
        if test_root is not None and test_root.exists():
            train_dataset = train_aug
        else:
            train_indices, _, test_indices = split_indices_by_class(train_eval.targets, 0.0, config.extra_test_split)
            train_dataset = Subset(train_aug, train_indices)
    else:
        t_test_split = 0.0 if (test_root is not None and test_root.exists()) else config.extra_test_split
        train_indices, val_indices, test_indices = split_indices_by_class(train_eval.targets, config.extra_val_split, t_test_split)
        if not val_indices:
            raise ValueError("Could not create validation split for extra dataset. Provide data/custom_incremental/val or add more images.")
        train_dataset = Subset(train_aug, train_indices)
        val_dataset = Subset(train_eval, val_indices)

    if test_root is not None and test_root.exists():
        test_dataset = ImageFolder(str(test_root), transform=test_transform)
    else:
        if val_root is not None and val_root.exists():
            _, _, test_indices = split_indices_by_class(train_eval.targets, 0.0, config.extra_test_split)
        test_dataset = Subset(train_eval, test_indices) if test_indices else val_dataset

    return {"name": source_name, "root": str(source_root), "train": train_dataset, "val": val_dataset, "test": test_dataset, "class_names": extra_class_names}


def load_coco_classification_split(split_root: Path, transform, source_name: str) -> SamplesDataset:
    annotations_path = split_root / "annotations.json"
    images_root = split_root / "images"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotations: {annotations_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Missing images folder: {images_root}")

    with open(annotations_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    categories = {
        int(item["id"]): canonicalize_source_label(source_name, item.get("name_readable") or item.get("name") or item["id"])
        for item in payload.get("categories", [])
    }
    image_names = {int(item["id"]): item["file_name"] for item in payload.get("images", []) if item.get("file_name")}

    best_ann: Dict[int, Tuple[float, int]] = {}
    for ann in payload.get("annotations", []):
        img_id, cat_id = int(ann["image_id"]), int(ann["category_id"])
        if cat_id not in categories:
            continue
        area = float(ann.get("area", 0.0))
        if img_id not in best_ann or area > best_ann[img_id][0]:
            best_ann[img_id] = (area, cat_id)

    labeled: List[Tuple[str, str]] = []
    for img_id, file_name in image_names.items():
        ann = best_ann.get(img_id)
        if ann is None:
            continue
        img_path = images_root / file_name
        if not img_path.exists() or not is_image_file(img_path):
            continue
        class_name = categories[ann[1]]
        if not should_keep_source_label(source_name, class_name):
            continue
        labeled.append((str(img_path), class_name))

    if not labeled:
        raise ValueError(f"No labeled samples found in {split_root}")

    class_names = sorted({cn for _, cn in labeled})
    c2i = {n: i for i, n in enumerate(class_names)}
    return SamplesDataset([(p, c2i[cn]) for p, cn in labeled], class_names, transform=transform)


def load_coco_extra_dataset(
    source_name: str, source_root: Path,
    train_root: Path, val_root: Optional[Path], test_root: Optional[Path],
    config: IncrementalConfig, train_transform, test_transform,
) -> Dict:
    train_aug = load_coco_classification_split(train_root, train_transform, source_name)
    train_eval = load_coco_classification_split(train_root, test_transform, source_name)
    extra_class_names = [normalize_label(n) for n in train_aug.classes]
    has_val = val_root is not None and (val_root / "annotations.json").exists()
    has_test = test_root is not None and (test_root / "annotations.json").exists()

    test_indices: List[int] = []
    val_indices: List[int] = []

    if has_val:
        val_dataset = load_coco_classification_split(val_root, test_transform, source_name)
        if has_test:
            train_dataset = train_aug
        else:
            train_indices, _, test_indices = split_indices_by_class(train_eval.targets, 0.0, config.extra_test_split)
            train_dataset = Subset(train_aug, train_indices)
    else:
        t_test = 0.0 if has_test else config.extra_test_split
        train_indices, val_indices, test_indices = split_indices_by_class(train_eval.targets, config.extra_val_split, t_test)
        if not val_indices:
            raise ValueError("Could not create validation split for COCO extra dataset.")
        train_dataset = Subset(train_aug, train_indices)
        val_dataset = Subset(train_eval, val_indices)

    if has_test:
        test_dataset = load_coco_classification_split(test_root, test_transform, source_name)
    else:
        if has_val:
            _, _, test_indices = split_indices_by_class(train_eval.targets, 0.0, config.extra_test_split)
        test_dataset = Subset(train_eval, test_indices) if test_indices else val_dataset

    return {"name": source_name, "root": str(source_root), "train": train_dataset, "val": val_dataset, "test": test_dataset, "class_names": extra_class_names}


def load_known_raw_extra_datasets(
    config: IncrementalConfig, train_transform, test_transform,
) -> Tuple[List[Dict], set[str]]:
    if not config.use_raw_known_sources:
        return [], set()

    sources: List[Dict] = []
    occupied: set[str] = set()
    for spec in KNOWN_RAW_SOURCES:
        source_root = find_existing_source_root(spec)
        if source_root is None:
            continue
        train_root = resolve_split_root_with_fallbacks(source_root, spec.get("train_subdir"), spec.get("fallback_train_subdirs"))
        if train_root is None:
            continue
        val_root = resolve_split_root(source_root, spec.get("val_subdir"))
        test_root = resolve_split_root(source_root, spec.get("test_subdir"))
        if val_root == train_root:
            val_root = None
        if test_root == train_root:
            test_root = None
        loader = spec.get("loader", "imagefolder")
        if loader == "coco_flat":
            loaded = load_coco_extra_dataset(spec["name"], source_root, train_root, val_root, test_root, config, train_transform, test_transform)
        else:
            loaded = load_extra_dataset_from_roots(spec["name"], source_root, train_root, val_root, test_root, config, train_transform, test_transform)
        sources.append(loaded)
        occupied.add(str(Path(spec["target_root"]).resolve()))

    return sources, occupied


def load_extra_datasets(config: IncrementalConfig, train_transform, test_transform) -> List[Dict]:
    raw_sources, occupied = load_known_raw_extra_datasets(config, train_transform, test_transform)
    sources = list(raw_sources)
    searched: List[str] = []

    for root_text in config.extra_data_dirs:
        extra_root = Path(root_text)
        searched.append(str(extra_root / "train"))
        if config.use_raw_known_sources and not config.include_extra_data_dirs_when_raw_available and str(extra_root.resolve()) in occupied:
            continue
        if not (extra_root / "train").exists():
            continue
        sources.append(load_extra_dataset_from_roots(
            extra_root.name, extra_root,
            extra_root / "train",
            extra_root / "val" if (extra_root / "val").exists() else None,
            extra_root / "test" if (extra_root / "test").exists() else None,
            config, train_transform, test_transform,
        ))

    excluded = [Path(config.data_dir) / "food-101"] + [Path(r) for r in config.extra_data_dirs]
    for spec in KNOWN_RAW_SOURCES:
        sr = find_existing_source_root(spec)
        if sr:
            excluded.append(sr)

    for item in discover_extra_data_roots(config, excluded):
        sources.append(load_extra_dataset_from_roots(
            str(item["name"]), Path(item["source_root"]), Path(item["train_root"]),
            Path(item["val_root"]) if item["val_root"] else None,
            Path(item["test_root"]) if item["test_root"] else None,
            config, train_transform, test_transform,
        ))

    if not sources:
        raise FileNotFoundError(
            "No extra incremental datasets were found. "
            f"Add a dataset with a train/ subfolder in: {', '.join(searched)}"
        )
    return sources


def load_food101_replay_datasets(
    config: IncrementalConfig, train_transform, test_transform,
) -> Tuple[Dataset, Dataset, Dataset, List[str]]:
    replay_train = Food101(root=config.data_dir, split="train", transform=train_transform, download=True)
    replay_eval = Food101(root=config.data_dir, split="test", transform=test_transform, download=False)

    train_idx = choose_subset_indices(len(replay_train), config.replay_train_samples)
    eval_idx = choose_subset_indices(len(replay_eval), config.replay_val_samples + config.replay_test_samples)

    replay_train_ds = Subset(replay_train, train_idx.tolist())
    val_count = min(config.replay_val_samples, len(eval_idx))
    replay_val_ds = Subset(replay_eval, eval_idx[:val_count].tolist())
    replay_test_ds = Subset(replay_eval, eval_idx[val_count:].tolist())
    if len(replay_test_ds) == 0:
        replay_test_ds = replay_val_ds

    class_names = [normalize_label(n) for n in replay_train.classes]
    return replay_train_ds, replay_val_ds, replay_test_ds, class_names


# ---------------------------------------------------------------------------
# Class merging and checkpoint expansion
# ---------------------------------------------------------------------------

def remap_dataset(dataset: Dataset, index_to_name: Sequence[str], class_to_index: Dict[str, int]) -> Dataset:
    return LabelRemapDataset(dataset, index_to_name=index_to_name, class_to_index=class_to_index)


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, config: TrainConfig) -> DataLoader:
    num_workers = max(0, int(config.num_workers))
    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0
    if os.name == "nt" and torch.cuda.is_available() and config.windows_stable_dataloader:
        num_workers = 0
        pin_memory = False
        persistent = False
    kwargs = {"num_workers": num_workers, "pin_memory": pin_memory, "persistent_workers": persistent}
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def load_expanded_checkpoint(
    model: torch.nn.Module,
    model_name: str,
    checkpoint_path: Path,
    base_class_names: Sequence[str],
    merged_class_names: Sequence[str],
) -> int:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state = model.state_dict()
    weight_key, bias_key = classifier_head_keys(model_name)

    compatible = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    model_state.update(compatible)

    classifier_weight = model_state[weight_key]
    classifier_bias = model_state[bias_key]
    old_weight = state_dict.get(weight_key)
    old_bias = state_dict.get(bias_key)
    merged_index = {name: i for i, name in enumerate(merged_class_names)}

    copied = 0
    if old_weight is not None and old_bias is not None:
        for old_i, class_name in enumerate(base_class_names):
            new_i = merged_index.get(class_name)
            if new_i is None:
                continue
            if old_i >= old_weight.shape[0] or new_i >= classifier_weight.shape[0]:
                continue
            classifier_weight[new_i].copy_(old_weight[old_i])
            classifier_bias[new_i].copy_(old_bias[old_i])
            copied += 1
        model_state[weight_key] = classifier_weight
        model_state[bias_key] = classifier_bias

    model.load_state_dict(model_state, strict=False)
    return copied


# ---------------------------------------------------------------------------
# Staging (raw datasets -> incremental folders)
# ---------------------------------------------------------------------------

def ensure_clean_split_root(target_root: Path, split_name: str, refresh: bool, cleared: set[str]) -> Path:
    split_root = target_root / split_name
    key = str(split_root)
    if refresh and split_root.exists() and key not in cleared:
        shutil.rmtree(split_root)
        cleared.add(key)
    split_root.mkdir(parents=True, exist_ok=True)
    return split_root


def stage_imagefolder_split(source_root: Path, target_root: Path, source_tag: str) -> int:
    staged = 0
    for class_dir in source_root.iterdir():
        if not class_dir.is_dir():
            continue
        target_class = target_root / normalize_label(class_dir.name)
        for img_path in class_dir.rglob("*"):
            if img_path.is_file() and is_image_file(img_path):
                dest_name = f"{normalize_label(source_tag)}__{img_path.name}"
                link_or_copy_file(img_path, target_class / dest_name)
                staged += 1
    return staged


def stage_known_raw_sources(config: IncrementalConfig) -> List[Dict]:
    if not config.auto_stage_known_sources:
        return []
    cleared: set[str] = set()
    staged: List[Dict] = []
    for spec in KNOWN_RAW_SOURCES:
        source_root = find_existing_source_root(spec)
        if source_root is None:
            continue
        target_root = Path(spec["target_root"])
        counts = {"train": 0, "val": 0, "test": 0}
        staged_any = False
        for src_key, split_name in (("train_subdir", "train"), ("val_subdir", "val"), ("test_subdir", "test")):
            split_root = resolve_split_root(source_root, spec.get(src_key))
            if split_root is None:
                continue
            target_split = ensure_clean_split_root(target_root, split_name, config.refresh_staged_sources, cleared)
            counts[split_name] = stage_imagefolder_split(split_root, target_split, spec["name"])
            staged_any = True
        if staged_any:
            staged.append({"name": spec["name"], "source_root": str(source_root), "target_root": str(target_root), **{f"{k}_files": v for k, v in counts.items()}})
    return staged


# ---------------------------------------------------------------------------
# Checkpoint promotion guard
# ---------------------------------------------------------------------------

def evaluate_promotion_decision(
    base_report: Dict, result: Dict, config: IncrementalConfig,
) -> Tuple[bool, List[str]]:
    if not config.protect_best_checkpoint:
        return True, ["checkpoint protection disabled"]
    base_metrics = base_report.get("best_model_metrics", {})
    reasons: List[str] = []
    old_top1 = base_metrics.get("test_top1_accuracy")
    old_top3 = base_metrics.get("test_top3_accuracy")
    if old_top1 is not None:
        drop = old_top1 - float(result["final_top1"])
        if drop > config.max_allowed_test_top1_drop:
            reasons.append(f"Top-1 drop {drop:.2f}% > allowed {config.max_allowed_test_top1_drop:.2f}%")
    if old_top3 is not None:
        drop = old_top3 - float(result["final_top3"])
        if drop > config.max_allowed_test_top3_drop:
            reasons.append(f"Top-3 drop {drop:.2f}% > allowed {config.max_allowed_test_top3_drop:.2f}%")
    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Dataloaders builder
# ---------------------------------------------------------------------------

def prepare_incremental_dataloaders(
    config: IncrementalConfig,
    train_config: TrainConfig,
    base_class_names: Sequence[str],
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict]:
    train_transform, test_transform = get_transforms(train_config.randaugment_magnitude)

    extra_sources = load_extra_datasets(config, train_transform, test_transform)
    replay_train, replay_val, replay_test, replay_class_names = load_food101_replay_datasets(
        config, train_transform, test_transform,
    )

    merged_class_names = list(base_class_names)
    merged_lookup = set(merged_class_names)
    extra_class_names_set: set[str] = set()
    for src in extra_sources:
        for cn in src["class_names"]:
            extra_class_names_set.add(cn)
            if cn not in merged_lookup:
                merged_class_names.append(cn)
                merged_lookup.add(cn)

    c2i = {n: i for i, n in enumerate(merged_class_names)}
    train_parts = [remap_dataset(replay_train, replay_class_names, c2i)]
    val_parts = [remap_dataset(replay_val, replay_class_names, c2i)]
    test_parts = [remap_dataset(replay_test, replay_class_names, c2i)]
    source_summaries: List[Dict] = []

    for src in extra_sources:
        train_parts.append(remap_dataset(src["train"], src["class_names"], c2i))
        val_parts.append(remap_dataset(src["val"], src["class_names"], c2i))
        test_parts.append(remap_dataset(src["test"], src["class_names"], c2i))
        source_summaries.append({
            "name": src["name"], "root": src["root"],
            "class_count": len(src["class_names"]),
            "train_samples": len(src["train"]),
            "val_samples": len(src["val"]),
            "test_samples": len(src["test"]),
        })

    train_ds = ConcatDataset(train_parts)
    val_ds = ConcatDataset(val_parts)
    test_ds = ConcatDataset(test_parts)

    extra_train = sum(s["train_samples"] for s in source_summaries)
    extra_val = sum(s["val_samples"] for s in source_summaries)
    extra_test = sum(s["test_samples"] for s in source_summaries)
    new_class_names = sorted(cn for cn in extra_class_names_set if cn not in set(base_class_names))

    data_summary = {
        "base_class_count": len(base_class_names),
        "extra_class_count": len(new_class_names),
        "new_class_names": new_class_names,
        "total_class_count": len(merged_class_names),
        "replay_train_samples": len(replay_train),
        "replay_val_samples": len(replay_val),
        "replay_test_samples": len(replay_test),
        "extra_train_samples": extra_train,
        "extra_val_samples": extra_val,
        "extra_test_samples": extra_test,
        "extra_sources": source_summaries,
        "train_samples_total": len(train_ds),
        "val_samples_total": len(val_ds),
        "test_samples_total": len(test_ds),
    }

    train_loader = build_loader(train_ds, train_config.batch_size, True, train_config)
    val_loader = build_loader(val_ds, train_config.batch_size, False, train_config)
    test_loader = build_loader(test_ds, train_config.eval_batch_size, False, train_config)
    return train_loader, val_loader, test_loader, merged_class_names, data_summary


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def load_base_report(config: IncrementalConfig) -> Dict:
    report_path = Path(config.runs_dir) / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Base report not found at {report_path}. Run baseline training first: python main.py train"
        )
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_base_class_names(config: IncrementalConfig, report: Dict) -> List[str]:
    if report.get("class_names"):
        return [normalize_label(n) for n in report["class_names"]]
    fallback = Food101(root=config.data_dir, split="test", download=False)
    return [normalize_label(n) for n in fallback.classes]


def backup_existing_artifacts(runs_dir: Path) -> None:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    for fn in ("best_model.pth", "report.json"):
        src = runs_dir / fn
        if src.exists():
            shutil.copyfile(src, runs_dir / f"{src.stem}_before_incremental_{stamp}{src.suffix}")


def _fmt_delta(new: float, old: Optional[float]) -> str:
    if old is None:
        return f"{new:.2f}%"
    d = new - old
    return f"{new:.2f}% ({'%+.2f' % d})"


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    start_time = time.time()
    inc_config = IncrementalConfig()

    base_report = load_base_report(inc_config)
    model_name = inc_config.model_name or base_report["best_model_name"]
    train_config = inc_config.build_train_config(model_name)

    set_seed(train_config.seed, enable_cudnn_benchmark=train_config.enable_cudnn_benchmark)
    log_environment_info()
    device = get_device()

    if device.type == "cuda":
        if os.name == "nt" and train_config.disable_tf32_on_windows_cuda:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        else:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    staged_sources = stage_known_raw_sources(inc_config)
    base_class_names = load_base_class_names(inc_config, base_report)
    train_loader, val_loader, test_loader, merged_class_names, data_summary = prepare_incremental_dataloaders(
        inc_config, train_config, base_class_names,
    )

    print("\n" + "=" * 60)
    print("INCREMENTAL FINE-TUNING")
    print("=" * 60)
    print(f"Base model: {model_name}")
    base_metrics = base_report.get("best_model_metrics", {})
    if base_metrics:
        print(f"Previous best: Top-1 {base_metrics.get('test_top1_accuracy', 0):.2f}% | Top-3 {base_metrics.get('test_top3_accuracy', 0):.2f}%")
    print(f"Classes: base={data_summary['base_class_count']} + new={data_summary['extra_class_count']} = total={data_summary['total_class_count']}")
    print(f"Training: replay={data_summary['replay_train_samples']} + extra={data_summary['extra_train_samples']} = {data_summary['train_samples_total']}")
    if data_summary.get("new_class_names"):
        preview = ", ".join(data_summary["new_class_names"][:15])
        print(f"New classes ({len(data_summary['new_class_names'])}): {preview}{' ...' if len(data_summary['new_class_names']) > 15 else ''}")

    model = build_model(model_name, num_classes=len(merged_class_names), pretrained=False)
    checkpoint_path = Path(inc_config.runs_dir) / "best_model.pth"
    copied = load_expanded_checkpoint(model, model_name, checkpoint_path, base_class_names, merged_class_names)
    print(f"Loaded {copied} classifier rows from previous checkpoint.")

    result = train_model(
        model, model_name, train_loader, val_loader, test_loader, train_config, device,
        comparison_metrics=base_metrics,
    )

    runs_dir = Path(inc_config.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = runs_dir / "best_model.pth"

    report = {
        "best_model_name": model_name,
        "training_mode": "incremental_finetune",
        "class_names": merged_class_names,
        "num_classes": len(merged_class_names),
        "base_report_timestamp": base_report.get("timestamp"),
        "best_model_metrics": {
            "val_top1_accuracy": result["best_val_top1"],
            "val_top3_accuracy": result["best_val_top3"],
            "test_top1_accuracy": result["final_top1"],
            "test_top3_accuracy": result["final_top3"],
        },
        "config": asdict(train_config),
        "incremental_config": asdict(inc_config),
        "data_summary": data_summary,
        "timestamp": datetime.now().isoformat(),
        "total_training_time_seconds": time.time() - start_time,
    }

    should_promote, rejection_reasons = evaluate_promotion_decision(base_report, result, inc_config)
    report["promotion_decision"] = {"promoted_to_best": should_promote, "rejection_reasons": rejection_reasons}

    if should_promote:
        if inc_config.backup_previous_best:
            backup_existing_artifacts(runs_dir)
        torch.save(result["model"].state_dict(), best_model_path)
        with open(runs_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nPromotion: updated best checkpoint and report.")
    else:
        with open(runs_dir / "last_incremental_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("\nPromotion: REJECTED — degradation threshold triggered.")
        for reason in rejection_reasons:
            print(f"  Reason: {reason}")

    old_top1 = base_metrics.get("test_top1_accuracy")
    old_top3 = base_metrics.get("test_top3_accuracy")
    print(f"\nTest Top-1: {_fmt_delta(result['final_top1'], old_top1)}")
    print(f"Test Top-3: {_fmt_delta(result['final_top3'], old_top3)}")
    print(f"Total time: {(time.time() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    main()
