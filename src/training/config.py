"""
Training configuration dataclasses.

Config           — Food-101 baseline training.
IncrementalConfig — Incremental fine-tuning on top of an existing checkpoint.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class Config:
    """Full configuration for the Food-101 baseline training pipeline."""

    # --- Data ---
    train_subset_size: int = 0      # 0 = full dataset (~75 k); 15000 = balanced; 2000 = quick test
    test_subset_size: int = 0       # 0 = full dataset (~25 k)
    val_split: float = 0.1          # Fraction of training split reserved for validation

    # --- Architecture ---
    models_to_train: Optional[List[str]] = None  # Set in __post_init__

    # --- Training phases ---
    warmup_epochs: int = 3          # Train only the classification head
    finetune_epochs: int = 10       # Fine-tune last blocks (0 = skip)

    # --- Optimisation ---
    batch_size: int = 32
    eval_batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    use_amp: bool = True
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0

    # --- Regularisation ---
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_augmentation_prob: float = 0.30
    randaugment_magnitude: int = 7
    use_ema: bool = True
    ema_decay: float = 0.9997
    eval_tta: bool = True
    val_tta: bool = False
    tta_num_views: int = 2

    # --- Training control ---
    early_stopping_patience: int = 5
    min_improvement_delta: float = 0.0
    fast_mode: bool = False
    run_eda: bool = True

    # --- Windows / GPU stability ---
    windows_stable_dataloader: bool = True
    skip_unstable_windows_cuda_models: bool = True
    force_vit_cpu: bool = False
    prioritize_gpu_stable_models: bool = True
    disable_tf32_on_windows_cuda: bool = True
    enable_cudnn_benchmark: bool = True

    # --- Paths ---
    seed: int = 42
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    runs_dir: str = "./runs"

    def __post_init__(self) -> None:
        if not (0.0 < self.val_split < 1.0):
            raise ValueError(f"val_split must be between 0 and 1, got {self.val_split}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.eval_batch_size < 1:
            raise ValueError(f"eval_batch_size must be >= 1, got {self.eval_batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.tta_num_views < 1:
            raise ValueError(f"tta_num_views must be >= 1, got {self.tta_num_views}")

        if self.models_to_train is None:
            self.models_to_train = ["resnet50", "efficientnet_b0"]

        if self.fast_mode:
            self.run_eda = False
            if len(self.models_to_train) > 1:
                self.models_to_train = [self.models_to_train[0]]

        is_windows = platform.system() == "Windows"
        has_cuda = torch.cuda.is_available()

        if has_cuda and is_windows and self.skip_unstable_windows_cuda_models:
            unstable = {"efficientnet_v2_s", "convnext_base"}
            removed = [m for m in self.models_to_train if m in unstable]
            if removed:
                self.models_to_train = [m for m in self.models_to_train if m not in unstable]
                print(f"Windows CUDA stability: skipping unstable models: {', '.join(removed)}")

        if has_cuda and is_windows and self.prioritize_gpu_stable_models:
            risky = {"vit_b_16"}
            stable_first = [m for m in self.models_to_train if m not in risky]
            risky_last = [m for m in self.models_to_train if m in risky]
            reordered = stable_first + risky_last
            if reordered != self.models_to_train:
                self.models_to_train = reordered
                print(f"Windows CUDA stability: reordered models: {', '.join(self.models_to_train)}")

        if has_cuda:
            heavy_models = {"vit_b_16", "resnet50", "efficientnet_b0"}
            has_heavy = any(m in heavy_models for m in self.models_to_train)
            try:
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                print(f"GPU VRAM detected: {total_vram:.2f} GB")
            except Exception:
                total_vram = None

            if self.batch_size <= 32:
                if total_vram is not None and total_vram >= 10.0:
                    self.batch_size = 64 if has_heavy else 128
                elif total_vram is not None and total_vram >= 6.0:
                    self.batch_size = 32 if has_heavy else 64
                else:
                    self.batch_size = 16 if has_heavy else 32

            if total_vram is not None and total_vram <= 4.0:
                self.batch_size = min(self.batch_size, 8 if has_heavy else 16)
                self.eval_batch_size = min(self.eval_batch_size, 8 if has_heavy else 16)
                self.use_ema = False
                self.eval_tta = False
                print("VRAM safety: reduced batch sizes, disabled EMA/TTA")

            if has_heavy:
                self.eval_batch_size = min(self.eval_batch_size, max(8, self.batch_size))
            elif self.eval_batch_size <= 16:
                self.eval_batch_size = max(self.batch_size, 32)

            if self.num_workers == 4:
                self.num_workers = 0 if is_windows else 8

            if is_windows and self.windows_stable_dataloader and self.enable_cudnn_benchmark:
                self.enable_cudnn_benchmark = False
                print("Windows CUDA stability: cuDNN benchmark disabled")

            print(
                f"GPU config: batch_size={self.batch_size}, "
                f"eval_batch_size={self.eval_batch_size}, "
                f"num_workers={self.num_workers}"
            )
            if self.use_amp:
                print("Mixed precision (AMP) enabled")
        else:
            if self.batch_size > 16:
                self.batch_size = 16
            if self.num_workers > 4:
                self.num_workers = 0 if is_windows else 4
            self.use_amp = False
            print(f"CPU config: batch_size={self.batch_size}, num_workers={self.num_workers}")


@dataclass
class IncrementalConfig:
    """Configuration for incremental fine-tuning on top of an existing checkpoint."""

    extra_data_dirs: List[str] = field(
        default_factory=lambda: ["./data/custom_incremental"]
    )
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    runs_dir: str = "./runs"
    model_name: str = ""

    # Food-101 replay to prevent catastrophic forgetting
    replay_train_samples: int = 4000
    replay_val_samples: int = 600
    replay_test_samples: int = 1200

    # Auto-split ratios when a dataset has no explicit val/test split
    extra_val_split: float = 0.15
    extra_test_split: float = 0.15

    # Training
    batch_size: int = 16
    eval_batch_size: int = 16
    warmup_epochs: int = 1
    finetune_epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42

    # Dataset handling
    auto_stage_known_sources: bool = False
    refresh_staged_sources: bool = False
    backup_previous_best: bool = False
    use_raw_known_sources: bool = True
    include_extra_data_dirs_when_raw_available: bool = False

    # Checkpoint protection
    protect_best_checkpoint: bool = True
    max_allowed_test_top1_drop: float = 3.0
    max_allowed_test_top3_drop: float = 2.0

    # Auto-discovery
    auto_discover_extra_data_dirs: bool = True
    discovery_max_depth: int = 4
    discovery_min_class_dirs: int = 2

    def build_train_config(self, model_name: str) -> "Config":
        return Config(
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
