"""
Incremental fine-tuning on top of the current best Food-101 checkpoint.

Expected extra dataset layout:
    data/custom_incremental/
        train/
            apple/
            banana/
        val/              # optional
        test/             # optional
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.datasets import Food101, ImageFolder

from src.train_food101 import Config as TrainConfig
from src.train_food101 import create_model, get_transforms, train_model
from src.utils import get_device, log_environment_info, set_seed


@dataclass
class IncrementalConfig:
    extra_data_dirs: List[str] = field(default_factory=lambda: [
        "./data/custom_incremental",
    ])
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    runs_dir: str = "./runs"
    model_name: str = ""
    replay_train_samples: int = 4000
    replay_val_samples: int = 600
    replay_test_samples: int = 1200
    extra_val_split: float = 0.15
    extra_test_split: float = 0.15
    batch_size: int = 16
    eval_batch_size: int = 16
    warmup_epochs: int = 1
    finetune_epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    auto_stage_known_sources: bool = False
    refresh_staged_sources: bool = False
    backup_previous_best: bool = False
    use_raw_known_sources: bool = True
    include_extra_data_dirs_when_raw_available: bool = False
    protect_best_checkpoint: bool = True
    max_allowed_test_top1_drop: float = 3.0
    max_allowed_test_top3_drop: float = 2.0
    auto_discover_extra_data_dirs: bool = True
    discovery_max_depth: int = 4
    discovery_min_class_dirs: int = 2

    def build_train_config(self, model_name: str) -> TrainConfig:
        return TrainConfig(
            train_subset_size=0,
            test_subset_size=0,
            batch_size=self.batch_size,
            eval_batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            models_to_train=[model_name],
            warmup_epochs=self.warmup_epochs,
            finetune_epochs=self.finetune_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            fast_mode=False,
            run_eda=False,
            seed=self.seed,
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            runs_dir=self.runs_dir,
        )


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


class LabelRemapDataset(Dataset):
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


def normalize_label(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


FRUITS_360_EXACT_LABEL_ALIASES = {
    "dangshan_pear": "pear",
    "bergamot_pear": "pear",
    "crown_pear": "pear",
    "hami_melon": "melon",
    "melon_piel_de_sapo": "melon",
    "cherry_tomato": "tomato",
    "red_currant": "currant",
    "black_currant": "currant",
}

FRUITS_360_PREFIX_ALIASES = {
    "apple": "apple",
    "pear": "pear",
    "tomato": "tomato",
    "pepper": "pepper",
    "potato": "potato",
    "onion": "onion",
    "garlic": "garlic",
    "cucumber": "cucumber",
    "banana": "banana",
    "orange": "orange",
    "grape": "grape",
    "mango": "mango",
    "peach": "peach",
    "nectarine": "nectarine",
    "plum": "plum",
    "melon": "melon",
    "kiwi": "kiwi",
    "lemon": "lemon",
    "lime": "lime",
    "apricot": "apricot",
    "carrot": "carrot",
    "cabbage": "cabbage",
    "cauliflower": "cauliflower",
    "papaya": "papaya",
    "pineapple": "pineapple",
    "pomegranate": "pomegranate",
    "radish": "radish",
    "strawberry": "strawberry",
    "watermelon": "watermelon",
}

FOOD_RECOGNITION_ALLOWED_TOKENS = {
    "apple", "apricot", "asparagus", "avocado", "banana", "beans", "beef", "beetroot",
    "blueberries", "bread", "broccoli", "burger", "cabbage", "cake", "carrot", "cashew",
    "cauliflower", "cheese", "cherry", "chicken", "chickpeas", "chips", "chocolate",
    "cod", "couscous", "crackers", "cucumber", "curry", "egg", "eggplant", "fajita",
    "fish", "french", "fruit", "garlic", "gnocchi", "grape", "ham", "hamburger",
    "halloumi", "hummus", "ice", "kiwi", "lamb", "lasagne", "leek", "lentils", "lettuce",
    "mandarine", "mango", "meat", "melon", "mozzarella", "mushroom", "mushrooms",
    "noodles", "nuts", "oats", "olive", "olives", "omelette", "onion", "orange", "pancakes",
    "pasta", "peanut", "pear", "peas", "penne", "pepper", "pizza", "pork", "potato",
    "potatoes", "pretzel", "pumpkin", "quiche", "radish", "ratatouille", "ravioli", "rice",
    "salad", "salami", "salmon", "sandwich", "sausage", "shrimp", "soup", "spinach",
    "spaghetti", "spring", "steak", "strawberries", "sushi", "taboule", "tart", "tomato",
    "tuna", "turkey", "vegetable", "vegetables", "veggie", "watermelon", "wrap", "zucchini",
}

FOOD_RECOGNITION_BLOCKED_TOKENS = {
    "alcohol", "aperitif", "beer", "bouillon", "butter", "caffeine", "cocktail", "coffee",
    "cola", "concentrate", "dressing", "drink", "espresso", "herbal", "ice_cubes", "juice",
    "ketchup", "lemonade", "liquor", "mayonnaise", "mineral", "oil", "powder", "prosecco",
    "ristretto", "salt", "sauce", "soda", "syrup", "tea", "vinegar", "water", "wine",
}

DISCOVERY_SKIP_DIR_NAMES = {
    "food_101",
    "custom_incremental",
    "fruitbot_expanded_incremental",
    "fruits_360_incremental",
    "uec_food_256_incremental",
    "__pycache__",
    "images",
    "meta",
    "hub",
    "raw_data",
    "chunks",
    "chunks_index",
    "tiles_index",
    "images_meta",
    "boxes",
    "areas",
    "masks",
    "iscrowds",
    "super_categories",
    "categories",
}

TRAIN_DIR_ALIASES = {"train", "training"}
VAL_DIR_ALIASES = {"val", "valid", "validation"}
TEST_DIR_ALIASES = {"test", "testing"}


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

    tokens = set(token for token in label.split("_") if token)
    if tokens & FOOD_RECOGNITION_BLOCKED_TOKENS:
        return False
    return bool(tokens & FOOD_RECOGNITION_ALLOWED_TOKENS)


def is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def looks_like_imagefolder_root(root: Path, min_class_dirs: int) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    class_dirs = [child for child in root.iterdir() if child.is_dir()]
    valid_class_dirs = 0
    for class_dir in class_dirs:
        has_image = any(image_path.is_file() and is_image_file(image_path) for image_path in class_dir.iterdir())
        if has_image:
            valid_class_dirs += 1
            if valid_class_dirs >= min_class_dirs:
                return True
    return False


def discover_extra_data_roots(
    config: IncrementalConfig,
    excluded_source_roots: Sequence[Path],
) -> List[Dict[str, Path | str | None]]:
    if not config.auto_discover_extra_data_dirs:
        return []

    data_root = Path(config.data_dir).resolve()
    excluded_roots = [root.resolve() for root in excluded_source_roots if root.exists()]
    discovered: List[Dict[str, Path | str | None]] = []
    seen_roots: set[str] = set()

    for current_root, dir_names, _ in os.walk(data_root, topdown=True):
        current_path = Path(current_root)
        relative_depth = len(current_path.relative_to(data_root).parts)
        if relative_depth > config.discovery_max_depth:
            dir_names[:] = []
            continue

        normalized_name = normalize_label(current_path.name)
        if normalized_name in DISCOVERY_SKIP_DIR_NAMES:
            dir_names[:] = []
            continue

        if any(is_under_root(current_path, excluded_root) for excluded_root in excluded_roots):
            dir_names[:] = []
            continue

        child_map = {
            normalize_label(child.name): child
            for child in current_path.iterdir()
            if child.is_dir()
        }
        train_root = next((child for name, child in child_map.items() if name in TRAIN_DIR_ALIASES), None)
        val_root = next((child for name, child in child_map.items() if name in VAL_DIR_ALIASES), None)
        test_root = next((child for name, child in child_map.items() if name in TEST_DIR_ALIASES), None)

        if train_root is not None and looks_like_imagefolder_root(train_root, config.discovery_min_class_dirs):
            resolved = str(current_path.resolve())
            if resolved not in seen_roots:
                discovered.append({
                    "name": normalize_label(current_path.name),
                    "source_root": current_path,
                    "train_root": train_root,
                    "val_root": val_root,
                    "test_root": test_root,
                })
                seen_roots.add(resolved)
            dir_names[:] = []
            continue

        parent_name = normalize_label(current_path.parent.name) if current_path.parent != current_path else ""
        if (
            parent_name not in TRAIN_DIR_ALIASES | VAL_DIR_ALIASES | TEST_DIR_ALIASES
            and looks_like_imagefolder_root(current_path, config.discovery_min_class_dirs)
        ):
            resolved = str(current_path.resolve())
            if resolved not in seen_roots:
                discovered.append({
                    "name": normalize_label(current_path.name),
                    "source_root": current_path,
                    "train_root": current_path,
                    "val_root": None,
                    "test_root": None,
                })
                seen_roots.add(resolved)
            dir_names[:] = []

    return discovered


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_existing_source_root(spec: Dict) -> Path | None:
    for root_text in spec["source_roots"]:
        root = Path(root_text)
        if root.exists():
            return root
    return None


def resolve_split_root(source_root: Path, split_name: str | None) -> Path | None:
    if split_name is None:
        return None
    if split_name == ".":
        return source_root
    split_root = source_root / split_name
    return split_root if split_root.exists() else None


def resolve_split_root_with_fallbacks(
    source_root: Path,
    split_name: str | None,
    fallback_split_names: Sequence[str] | None = None,
) -> Path | None:
    split_root = resolve_split_root(source_root, split_name)
    if split_root is not None:
        return split_root

    for fallback_name in fallback_split_names or []:
        split_root = resolve_split_root(source_root, fallback_name)
        if split_root is not None:
            return split_root

    return None


def ensure_clean_split_root(target_root: Path, split_name: str, refresh: bool, cleared_roots: set[str]) -> Path:
    split_root = target_root / split_name
    key = str(split_root)
    if refresh and split_root.exists() and key not in cleared_roots:
        shutil.rmtree(split_root)
        cleared_roots.add(key)
    split_root.mkdir(parents=True, exist_ok=True)
    return split_root


def link_or_copy_file(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage_imagefolder_split(source_root: Path, target_root: Path, source_tag: str) -> int:
    staged_files = 0
    class_directories = [path for path in source_root.iterdir() if path.is_dir()]
    for class_dir in class_directories:
        target_class_dir = target_root / normalize_label(class_dir.name)
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and is_image_file(image_path):
                destination_name = f"{normalize_label(source_tag)}__{image_path.name}"
                link_or_copy_file(image_path, target_class_dir / destination_name)
                staged_files += 1
    return staged_files


def stage_known_raw_sources(config: IncrementalConfig) -> List[Dict]:
    if not config.auto_stage_known_sources:
        return []

    cleared_roots: set[str] = set()
    staged_sources: List[Dict] = []
    for spec in KNOWN_RAW_SOURCES:
        source_root = find_existing_source_root(spec)
        if source_root is None:
            continue

        target_root = Path(spec["target_root"])
        staged_any = False
        split_counts = {"train": 0, "val": 0, "test": 0}
        for source_key, target_split in (
            ("train_subdir", "train"),
            ("val_subdir", "val"),
            ("test_subdir", "test"),
        ):
            split_root = resolve_split_root(source_root, spec.get(source_key))
            if split_root is None or not split_root.exists():
                continue
            target_split_root = ensure_clean_split_root(
                target_root,
                target_split,
                config.refresh_staged_sources,
                cleared_roots,
            )
            split_counts[target_split] = stage_imagefolder_split(split_root, target_split_root, spec["name"])
            staged_any = True

        if staged_any:
            staged_sources.append({
                "name": spec["name"],
                "source_root": str(source_root),
                "target_root": str(target_root),
                "train_files": split_counts["train"],
                "val_files": split_counts["val"],
                "test_files": split_counts["test"],
            })

    if staged_sources:
        print("Staged raw datasets into incremental folders:")
        for item in staged_sources:
            print(
                f"  - {item['name']}: train={item['train_files']}, "
                f"val={item['val_files']}, test={item['test_files']}"
            )

    return staged_sources


def choose_subset_indices(total_size: int, requested_size: int) -> np.ndarray:
    if requested_size <= 0 or requested_size >= total_size:
        return np.arange(total_size)
    return np.random.choice(total_size, requested_size, replace=False)


def split_indices_by_class(
    targets: Sequence[int],
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    class_to_indices: Dict[int, List[int]] = {}
    for index, target in enumerate(targets):
        class_to_indices.setdefault(int(target), []).append(index)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for class_indices in class_to_indices.values():
        shuffled = np.array(class_indices, dtype=np.int64)
        np.random.shuffle(shuffled)
        total = len(shuffled)
        max_holdout = max(0, total - 1)

        val_count = min(int(round(total * val_ratio)), max_holdout)
        remaining_holdout = max_holdout - val_count
        test_count = min(int(round(total * test_ratio)), remaining_holdout)

        if val_ratio > 0 and val_count == 0 and total >= 3:
            val_count = 1
            remaining_holdout = max(0, total - 1 - val_count)
            test_count = min(test_count, remaining_holdout)

        if test_ratio > 0 and test_count == 0 and total - val_count >= 2 and remaining_holdout > 0:
            test_count = 1

        val_indices.extend(shuffled[:val_count].tolist())
        test_start = val_count
        test_end = val_count + test_count
        test_indices.extend(shuffled[test_start:test_end].tolist())
        train_indices.extend(shuffled[test_end:].tolist())

    return train_indices, val_indices, test_indices


def load_base_report(config: IncrementalConfig) -> Dict:
    report_path = Path(config.runs_dir) / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Missing base report at {report_path}. Train Food-101 first with python -m src.train_food101."
        )
    with open(report_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_base_class_names(config: IncrementalConfig, report: Dict) -> List[str]:
    if report.get("class_names"):
        return [normalize_label(name) for name in report["class_names"]]

    fallback_dataset = Food101(root=config.data_dir, split="test", download=False)
    return [normalize_label(name) for name in fallback_dataset.classes]


def load_single_extra_dataset(
    extra_root: Path,
    config: IncrementalConfig,
    train_transform,
    test_transform,
) -> Dict:
    return load_extra_dataset_from_roots(
        source_name=extra_root.name,
        source_root=extra_root,
        train_root=extra_root / "train",
        val_root=extra_root / "val",
        test_root=extra_root / "test",
        config=config,
        train_transform=train_transform,
        test_transform=test_transform,
    )


def load_extra_dataset_from_roots(
    source_name: str,
    source_root: Path,
    train_root: Path,
    val_root: Path | None,
    test_root: Path | None,
    config: IncrementalConfig,
    train_transform,
    test_transform,
) -> Dict:
    train_dataset_aug = ImageFolder(str(train_root), transform=train_transform)
    train_dataset_eval = ImageFolder(str(train_root), transform=test_transform)
    extra_class_names = [canonicalize_source_label(source_name, name) for name in train_dataset_aug.classes]
    test_indices: List[int] = []
    val_indices: List[int] = []

    if val_root is not None and val_root.exists():
        val_dataset = ImageFolder(str(val_root), transform=test_transform)
        if test_root is not None and test_root.exists():
            train_dataset = train_dataset_aug
        else:
            train_indices, _, test_indices = split_indices_by_class(
                train_dataset_eval.targets,
                0.0,
                config.extra_test_split,
            )
            train_dataset = Subset(train_dataset_aug, train_indices)
    else:
        train_indices, val_indices, test_indices = split_indices_by_class(
            train_dataset_eval.targets,
            config.extra_val_split,
            0.0 if (test_root is not None and test_root.exists()) else config.extra_test_split,
        )
        if not val_indices:
            raise ValueError(
                "Could not create a validation split for the extra dataset. Add more images or provide data/custom_incremental/val."
            )
        train_dataset = Subset(train_dataset_aug, train_indices)
        val_dataset = Subset(train_dataset_eval, val_indices)

    if test_root is not None and test_root.exists():
        test_dataset = ImageFolder(str(test_root), transform=test_transform)
    else:
        if val_root is not None and val_root.exists():
            _, _, test_indices = split_indices_by_class(
                train_dataset_eval.targets,
                0.0,
                config.extra_test_split,
            )
        if not test_indices:
            if val_root is not None and val_root.exists():
                test_dataset = val_dataset
            else:
                test_indices = val_indices
                test_dataset = Subset(train_dataset_eval, test_indices)
        else:
            test_dataset = Subset(train_dataset_eval, test_indices)

    return {
        "name": source_name,
        "root": str(source_root),
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "class_names": extra_class_names,
    }


def load_coco_classification_split(split_root: Path, transform, source_name: str) -> SamplesDataset:
    annotations_path = split_root / "annotations.json"
    images_root = split_root / "images"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotations file: {annotations_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Missing images folder: {images_root}")

    with open(annotations_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    categories = {
        int(item["id"]): canonicalize_source_label(
            source_name,
            item.get("name_readable") or item.get("name") or item["id"],
        )
        for item in payload.get("categories", [])
    }
    image_names = {
        int(item["id"]): item["file_name"]
        for item in payload.get("images", [])
        if item.get("file_name")
    }

    best_annotation_by_image: Dict[int, Tuple[float, int]] = {}
    for annotation in payload.get("annotations", []):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if category_id not in categories:
            continue
        area = float(annotation.get("area", 0.0))
        previous = best_annotation_by_image.get(image_id)
        if previous is None or area > previous[0]:
            best_annotation_by_image[image_id] = (area, category_id)

    labeled_samples: List[Tuple[str, str]] = []
    for image_id, file_name in image_names.items():
        best_annotation = best_annotation_by_image.get(image_id)
        if best_annotation is None:
            continue
        image_path = images_root / file_name
        if not image_path.exists() or not is_image_file(image_path):
            continue
        class_name = categories[best_annotation[1]]
        if not should_keep_source_label(source_name, class_name):
            continue
        labeled_samples.append((str(image_path), class_name))

    if not labeled_samples:
        raise ValueError(f"No labeled image samples found in {split_root}")

    class_names = sorted({class_name for _, class_name in labeled_samples})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    indexed_samples = [(image_path, class_to_index[class_name]) for image_path, class_name in labeled_samples]
    return SamplesDataset(indexed_samples, class_names, transform=transform)


def has_coco_annotations(split_root: Path | None) -> bool:
    return split_root is not None and (split_root / "annotations.json").exists()


def load_coco_extra_dataset(
    source_name: str,
    source_root: Path,
    train_root: Path,
    val_root: Path | None,
    test_root: Path | None,
    config: IncrementalConfig,
    train_transform,
    test_transform,
) -> Dict:
    train_dataset_aug = load_coco_classification_split(train_root, train_transform, source_name)
    train_dataset_eval = load_coco_classification_split(train_root, test_transform, source_name)
    extra_class_names = [normalize_label(name) for name in train_dataset_aug.classes]
    test_indices: List[int] = []
    val_indices: List[int] = []
    has_labeled_val = has_coco_annotations(val_root)
    has_labeled_test = has_coco_annotations(test_root)

    if has_labeled_val:
        val_dataset = load_coco_classification_split(val_root, test_transform, source_name)
        if has_labeled_test:
            train_dataset = train_dataset_aug
        else:
            train_indices, _, test_indices = split_indices_by_class(
                train_dataset_eval.targets,
                0.0,
                config.extra_test_split,
            )
            train_dataset = Subset(train_dataset_aug, train_indices)
    else:
        train_indices, val_indices, test_indices = split_indices_by_class(
            train_dataset_eval.targets,
            config.extra_val_split,
            0.0 if (test_root is not None and test_root.exists()) else config.extra_test_split,
        )
        if not val_indices:
            raise ValueError(
                "Could not create a validation split for the COCO-style extra dataset. Add more images or provide an explicit val split."
            )
        train_dataset = Subset(train_dataset_aug, train_indices)
        val_dataset = Subset(train_dataset_eval, val_indices)

    if has_labeled_test:
        test_dataset = load_coco_classification_split(test_root, test_transform, source_name)
    else:
        if has_labeled_val:
            _, _, test_indices = split_indices_by_class(
                train_dataset_eval.targets,
                0.0,
                config.extra_test_split,
            )
        if not test_indices:
            if has_labeled_val:
                test_dataset = val_dataset
            else:
                test_indices = val_indices
                test_dataset = Subset(train_dataset_eval, test_indices)
        else:
            test_dataset = Subset(train_dataset_eval, test_indices)

    return {
        "name": source_name,
        "root": str(source_root),
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "class_names": extra_class_names,
    }


def load_known_raw_extra_datasets(
    config: IncrementalConfig,
    train_transform,
    test_transform,
) -> Tuple[List[Dict], set[str]]:
    if not config.use_raw_known_sources:
        return [], set()

    sources: List[Dict] = []
    occupied_incremental_roots: set[str] = set()
    for spec in KNOWN_RAW_SOURCES:
        source_root = find_existing_source_root(spec)
        if source_root is None:
            continue

        train_root = resolve_split_root_with_fallbacks(
            source_root,
            spec.get("train_subdir"),
            spec.get("fallback_train_subdirs"),
        )
        if train_root is None or not train_root.exists():
            continue

        val_root = resolve_split_root(source_root, spec.get("val_subdir"))
        test_root = resolve_split_root(source_root, spec.get("test_subdir"))
        if val_root is not None and val_root == train_root:
            val_root = None
        if test_root is not None and test_root == train_root:
            test_root = None
        loader_name = spec.get("loader", "imagefolder")
        if loader_name == "coco_flat":
            loaded_source = load_coco_extra_dataset(
                source_name=spec["name"],
                source_root=source_root,
                train_root=train_root,
                val_root=val_root,
                test_root=test_root,
                config=config,
                train_transform=train_transform,
                test_transform=test_transform,
            )
        else:
            loaded_source = load_extra_dataset_from_roots(
                source_name=spec["name"],
                source_root=source_root,
                train_root=train_root,
                val_root=val_root,
                test_root=test_root,
                config=config,
                train_transform=train_transform,
                test_transform=test_transform,
            )
        sources.append(loaded_source)
        occupied_incremental_roots.add(str(Path(spec["target_root"]).resolve()))

    return sources, occupied_incremental_roots


def load_extra_datasets(
    config: IncrementalConfig,
    train_transform,
    test_transform,
) -> List[Dict]:
    raw_sources, occupied_incremental_roots = load_known_raw_extra_datasets(
        config,
        train_transform,
        test_transform,
    )
    sources: List[Dict] = list(raw_sources)
    searched_paths = []
    for root_text in config.extra_data_dirs:
        extra_root = Path(root_text)
        searched_paths.append(str(extra_root / "train"))
        if (
            config.use_raw_known_sources
            and not config.include_extra_data_dirs_when_raw_available
            and str(extra_root.resolve()) in occupied_incremental_roots
        ):
            continue
        if not (extra_root / "train").exists():
            continue
        sources.append(load_single_extra_dataset(extra_root, config, train_transform, test_transform))

    excluded_source_roots = [Path(config.data_dir) / "food-101"]
    excluded_source_roots.extend(Path(root_text) for root_text in config.extra_data_dirs)
    for spec in KNOWN_RAW_SOURCES:
        source_root = find_existing_source_root(spec)
        if source_root is not None:
            excluded_source_roots.append(source_root)

    discovered_roots = discover_extra_data_roots(config, excluded_source_roots)
    for item in discovered_roots:
        sources.append(
            load_extra_dataset_from_roots(
                source_name=str(item["name"]),
                source_root=Path(item["source_root"]),
                train_root=Path(item["train_root"]),
                val_root=Path(item["val_root"]) if item["val_root"] is not None else None,
                test_root=Path(item["test_root"]) if item["test_root"] is not None else None,
                config=config,
                train_transform=train_transform,
                test_transform=test_transform,
            )
        )

    if not sources:
        searched_text = ", ".join(searched_paths)
        raise FileNotFoundError(
            "No extra incremental datasets were found. Add at least one dataset with a train/ folder in: "
            f"{searched_text}"
        )

    return sources


def load_food101_replay_datasets(
    config: IncrementalConfig,
    train_transform,
    test_transform,
) -> Tuple[Dataset, Dataset, Dataset, List[str]]:
    replay_train_dataset = Food101(
        root=config.data_dir,
        split="train",
        transform=train_transform,
        download=True,
    )
    replay_eval_dataset = Food101(
        root=config.data_dir,
        split="test",
        transform=test_transform,
        download=False,
    )

    train_indices = choose_subset_indices(len(replay_train_dataset), config.replay_train_samples)
    eval_indices = choose_subset_indices(
        len(replay_eval_dataset),
        config.replay_val_samples + config.replay_test_samples,
    )

    replay_train = Subset(replay_train_dataset, train_indices.tolist())
    replay_val_count = min(config.replay_val_samples, len(eval_indices))
    replay_val = Subset(replay_eval_dataset, eval_indices[:replay_val_count].tolist())
    replay_test = Subset(replay_eval_dataset, eval_indices[replay_val_count:].tolist())
    if len(replay_test) == 0:
        replay_test = replay_val

    class_names = [normalize_label(name) for name in replay_train_dataset.classes]
    return replay_train, replay_val, replay_test, class_names


def remap_dataset(dataset: Dataset, index_to_name: Sequence[str], class_to_index: Dict[str, int]) -> Dataset:
    return LabelRemapDataset(dataset, index_to_name=index_to_name, class_to_index=class_to_index)


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, config: TrainConfig) -> DataLoader:
    loader_num_workers = max(0, int(config.num_workers))
    use_pin_memory = torch.cuda.is_available()
    use_persistent_workers = loader_num_workers > 0

    if os.name == "nt" and torch.cuda.is_available() and config.windows_stable_dataloader:
        loader_num_workers = 0
        use_pin_memory = False
        use_persistent_workers = False

    loader_kwargs = {
        "num_workers": loader_num_workers,
        "pin_memory": use_pin_memory,
        "persistent_workers": use_persistent_workers,
    }
    if loader_num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **loader_kwargs,
    )


def classifier_state_keys(model_name: str) -> Tuple[str, str]:
    if model_name == "resnet50":
        return "fc.weight", "fc.bias"
    if model_name in {"efficientnet_b0", "efficientnet_b4", "efficientnet_v2_s", "mobilenet_v3_large"}:
        return "classifier.1.weight", "classifier.1.bias"
    if model_name == "convnext_base":
        return "classifier.2.weight", "classifier.2.bias"
    if model_name == "vit_b_16":
        return "heads.head.weight", "heads.head.bias"
    raise ValueError(f"Unsupported model for incremental loading: {model_name}")


def load_expanded_checkpoint(
    model: torch.nn.Module,
    model_name: str,
    checkpoint_path: Path,
    base_class_names: Sequence[str],
    merged_class_names: Sequence[str],
) -> int:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state = model.state_dict()
    weight_key, bias_key = classifier_state_keys(model_name)

    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model_state.update(compatible)

    classifier_weight = model_state[weight_key]
    classifier_bias = model_state[bias_key]
    old_weight = state_dict.get(weight_key)
    old_bias = state_dict.get(bias_key)
    merged_index = {name: index for index, name in enumerate(merged_class_names)}

    copied_rows = 0
    if old_weight is not None and old_bias is not None:
        for old_index, class_name in enumerate(base_class_names):
            new_index = merged_index.get(class_name)
            if new_index is None:
                continue
            if old_index >= old_weight.shape[0] or new_index >= classifier_weight.shape[0]:
                continue
            classifier_weight[new_index].copy_(old_weight[old_index])
            classifier_bias[new_index].copy_(old_bias[old_index])
            copied_rows += 1

        model_state[weight_key] = classifier_weight
        model_state[bias_key] = classifier_bias

    model.load_state_dict(model_state, strict=False)
    return copied_rows


def backup_existing_artifacts(runs_dir: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    for filename in ("best_model.pth", "report.json"):
        source = runs_dir / filename
        if source.exists():
            backup_name = f"{source.stem}_before_incremental_{timestamp}{source.suffix}"
            shutil.copyfile(source, runs_dir / backup_name)


def prepare_incremental_dataloaders(
    config: IncrementalConfig,
    train_config: TrainConfig,
    base_class_names: Sequence[str],
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict]:
    train_transform, test_transform = get_transforms(train_config)

    extra_sources = load_extra_datasets(
        config,
        train_transform,
        test_transform,
    )
    replay_train, replay_val, replay_test, replay_class_names = load_food101_replay_datasets(
        config,
        train_transform,
        test_transform,
    )

    merged_class_names = list(base_class_names)
    merged_lookup = set(merged_class_names)
    extra_class_names_set = set()
    for source in extra_sources:
        for class_name in source["class_names"]:
            extra_class_names_set.add(class_name)
            if class_name not in merged_lookup:
                merged_class_names.append(class_name)
                merged_lookup.add(class_name)

    class_to_index = {name: index for index, name in enumerate(merged_class_names)}
    train_parts = [remap_dataset(replay_train, replay_class_names, class_to_index)]
    val_parts = [remap_dataset(replay_val, replay_class_names, class_to_index)]
    test_parts = [remap_dataset(replay_test, replay_class_names, class_to_index)]

    source_summaries = []
    extra_train_samples = 0
    extra_val_samples = 0
    extra_test_samples = 0
    for source in extra_sources:
        train_parts.append(remap_dataset(source["train"], source["class_names"], class_to_index))
        val_parts.append(remap_dataset(source["val"], source["class_names"], class_to_index))
        test_parts.append(remap_dataset(source["test"], source["class_names"], class_to_index))
        extra_train_samples += len(source["train"])
        extra_val_samples += len(source["val"])
        extra_test_samples += len(source["test"])
        source_summaries.append({
            "name": source["name"],
            "root": source["root"],
            "class_count": len(source["class_names"]),
            "train_samples": len(source["train"]),
            "val_samples": len(source["val"]),
            "test_samples": len(source["test"]),
        })

    train_dataset = ConcatDataset(train_parts)
    val_dataset = ConcatDataset(val_parts)
    test_dataset = ConcatDataset(test_parts)

    data_summary = {
        "base_class_count": len(base_class_names),
        "extra_class_count": len([name for name in extra_class_names_set if name not in set(base_class_names)]),
        "new_class_names": sorted([name for name in extra_class_names_set if name not in set(base_class_names)]),
        "total_class_count": len(merged_class_names),
        "replay_train_samples": len(replay_train),
        "replay_val_samples": len(replay_val),
        "replay_test_samples": len(replay_test),
        "extra_train_samples": extra_train_samples,
        "extra_val_samples": extra_val_samples,
        "extra_test_samples": extra_test_samples,
        "extra_sources": source_summaries,
        "train_samples_total": len(train_dataset),
        "val_samples_total": len(val_dataset),
        "test_samples_total": len(test_dataset),
    }

    train_loader = build_loader(train_dataset, train_config.batch_size, True, train_config)
    val_loader = build_loader(val_dataset, train_config.batch_size, False, train_config)
    test_loader = build_loader(test_dataset, train_config.eval_batch_size, False, train_config)
    return train_loader, val_loader, test_loader, merged_class_names, data_summary


def format_metric_delta(new_value: float, old_value: float | None) -> str:
    if old_value is None:
        return f"{new_value:.2f}%"
    delta = new_value - old_value
    sign = "+" if delta >= 0 else ""
    return f"{new_value:.2f}% ({sign}{delta:.2f})"


def print_incremental_overview(
    model_name: str,
    base_report: Dict,
    data_summary: Dict,
    staged_sources: List[Dict],
) -> None:
    print("\n" + "=" * 60)
    print("INCREMENTAL FINE-TUNING")
    print("=" * 60)
    print(f"Base model: {model_name}")

    base_metrics = base_report.get("best_model_metrics", {})
    if base_metrics:
        print(
            "Previous best metrics: "
            f"Top-1 {base_metrics.get('test_top1_accuracy', 0):.2f}% | "
            f"Top-3 {base_metrics.get('test_top3_accuracy', 0):.2f}%"
        )
        print(
            "Improvement target: "
            f"Val Top-1 {base_metrics.get('val_top1_accuracy', 0):.2f}% | "
            f"Val Top-3 {base_metrics.get('val_top3_accuracy', 0):.2f}% | "
            f"Test Top-1 {base_metrics.get('test_top1_accuracy', 0):.2f}% | "
            f"Test Top-3 {base_metrics.get('test_top3_accuracy', 0):.2f}%"
        )

    print(
        f"Class merge: base={data_summary['base_class_count']} | "
        f"new={data_summary['extra_class_count']} | total={data_summary['total_class_count']}"
    )
    print(
        f"Training samples: replay={data_summary['replay_train_samples']} + "
        f"extra={data_summary['extra_train_samples']} = total={data_summary['train_samples_total']}"
    )
    print(
        f"Validation samples: replay={data_summary['replay_val_samples']} + "
        f"extra={data_summary['extra_val_samples']} = total={data_summary['val_samples_total']}"
    )
    print(
        f"Test samples: replay={data_summary['replay_test_samples']} + "
        f"extra={data_summary['extra_test_samples']} = total={data_summary['test_samples_total']}"
    )

    if staged_sources:
        print("Staged raw sources:")
        for item in staged_sources:
            print(
                f"  - {item['name']} -> {item['target_root']} "
                f"(train={item['train_files']}, val={item['val_files']}, test={item['test_files']})"
            )

    if data_summary["extra_sources"]:
        print("Active training sources:")
        for source in data_summary["extra_sources"]:
            print(
                f"  - {source['name']}: classes={source['class_count']}, "
                f"train={source['train_samples']}, val={source['val_samples']}, test={source['test_samples']}"
            )

    new_classes = data_summary.get("new_class_names", [])
    if new_classes:
        preview = ", ".join(new_classes[:15])
        suffix = " ..." if len(new_classes) > 15 else ""
        print(f"New classes added ({len(new_classes)}): {preview}{suffix}")


def evaluate_promotion_decision(
    base_report: Dict,
    result: Dict,
    config: IncrementalConfig,
) -> Tuple[bool, List[str]]:
    if not config.protect_best_checkpoint:
        return True, ["best checkpoint protection disabled"]

    base_metrics = base_report.get("best_model_metrics", {})
    old_top1 = base_metrics.get("test_top1_accuracy")
    old_top3 = base_metrics.get("test_top3_accuracy")
    reasons: List[str] = []

    if old_top1 is not None:
        top1_drop = old_top1 - float(result["final_top1"])
        if top1_drop > config.max_allowed_test_top1_drop:
            reasons.append(
                f"test_top1 drop {top1_drop:.2f} exceeded {config.max_allowed_test_top1_drop:.2f}"
            )

    if old_top3 is not None:
        top3_drop = old_top3 - float(result["final_top3"])
        if top3_drop > config.max_allowed_test_top3_drop:
            reasons.append(
                f"test_top3 drop {top3_drop:.2f} exceeded {config.max_allowed_test_top3_drop:.2f}"
            )

    return len(reasons) == 0, reasons


def main() -> None:
    start_time = time.time()
    incremental_config = IncrementalConfig()
    base_report = load_base_report(incremental_config)
    model_name = incremental_config.model_name or base_report["best_model_name"]
    train_config = incremental_config.build_train_config(model_name)

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

    staged_sources = stage_known_raw_sources(incremental_config)
    base_class_names = load_base_class_names(incremental_config, base_report)
    train_loader, val_loader, test_loader, merged_class_names, data_summary = prepare_incremental_dataloaders(
        incremental_config,
        train_config,
        base_class_names,
    )
    print_incremental_overview(model_name, base_report, data_summary, staged_sources)

    model = create_model(model_name, num_classes=len(merged_class_names), pretrained=False)
    checkpoint_path = Path(incremental_config.runs_dir) / "best_model.pth"
    copied_rows = load_expanded_checkpoint(
        model,
        model_name,
        checkpoint_path,
        base_class_names=base_class_names,
        merged_class_names=merged_class_names,
    )
    print(f"Loaded checkpoint rows from previous head: {copied_rows}")

    result = train_model(
        model,
        model_name,
        train_loader,
        val_loader,
        test_loader,
        train_config,
        device,
        comparison_metrics=base_report.get("best_model_metrics", {}),
    )

    runs_dir = Path(incremental_config.runs_dir)
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
        "incremental_config": asdict(incremental_config),
        "data_summary": data_summary,
        "timestamp": datetime.now().isoformat(),
        "total_training_time_seconds": time.time() - start_time,
    }
    should_promote, rejection_reasons = evaluate_promotion_decision(
        base_report,
        result,
        incremental_config,
    )
    report["promotion_decision"] = {
        "promoted_to_best": should_promote,
        "rejection_reasons": rejection_reasons,
    }

    if should_promote:
        if incremental_config.backup_previous_best:
            backup_existing_artifacts(runs_dir)
        torch.save(result["model"].state_dict(), best_model_path)
        with open(runs_dir / "report.json", "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
    else:
        with open(runs_dir / "last_incremental_report.json", "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

    old_top1 = base_report.get("best_model_metrics", {}).get("test_top1_accuracy")
    old_top3 = base_report.get("best_model_metrics", {}).get("test_top3_accuracy")
    print("\nTraining result:")
    print(f"  Test Top-1: {format_metric_delta(result['final_top1'], old_top1)}")
    print(f"  Test Top-3: {format_metric_delta(result['final_top3'], old_top3)}")
    print(f"  Validation Top-1: {result['best_val_top1']:.2f}%")
    print(f"  Validation Top-3: {result['best_val_top3']:.2f}%")
    if should_promote:
        print(f"  Replaced best checkpoint: {best_model_path}")
        print(f"  Updated report: {runs_dir / 'report.json'}")
    else:
        print("  Kept existing best checkpoint: degradation threshold triggered")
        for reason in rejection_reasons:
            print(f"  Reject reason: {reason}")
        print(f"  Saved run report: {runs_dir / 'last_incremental_report.json'}")


if __name__ == "__main__":
    main()
