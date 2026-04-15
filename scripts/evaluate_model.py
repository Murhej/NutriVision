"""
Comprehensive model evaluation — per-class accuracy, confusion analysis,
confidence statistics, and class-group comparison.

Works for any checkpoint: Food-101 baseline (101 classes) or any incremental
model with extra classes.  Reads runs/report.json to know which extra data
sources were used and reconstructs the test set automatically.

Outputs (all in --out-dir, default outputs/):
    eval_per_class.png    — per-class Top-1/Top-3 accuracy bar chart
    eval_confusion.png    — full confusion heat-map + zoomed worst-class block
    eval_confidence.png   — confidence histograms, calibration curve, scatter
    eval_groups.png       — class-group accuracy comparison
    eval_metrics.json     — all computed numbers (reuse with --cached)

Usage:
    python scripts/evaluate_model.py                         # full eval
    python scripts/evaluate_model.py --model runs/best_model.pth
    python scripts/evaluate_model.py --cached                # skip inference
    python scripts/evaluate_model.py --sample 300           # 300 imgs/class
    python scripts/evaluate_model.py --out-dir outputs/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

RUNS_DIR   = BASE_DIR / "runs"
DATA_DIR   = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

DARK_BG   = "#0f0f0f"
PANEL_BG  = "#1a1a1a"
GRID_COL  = "#2e2e2e"
TEXT_COL  = "#cccccc"
DIM_TEXT  = "#888888"
C_FOOD101 = "#4c9be8"   # blue  — Food-101 base classes
C_NEW     = "#34c47c"   # green — incremental new classes
C_TOP3    = "#f0a500"   # amber — top-3 accent
C_BAD     = "#e84c4c"   # red   — worst performers


# ─────────────────────────────────────────────────────────────────────────────
# Dataset utilities
# ─────────────────────────────────────────────────────────────────────────────

class _RemappedDataset(Dataset):
    """Wrap a dataset and remap its integer labels.  Samples whose label has
    no entry in *index_map* are silently dropped when *skip_unmapped=True*."""

    def __init__(self, base: Dataset, index_map: Dict[int, int],
                 skip_unmapped: bool = True):
        self.base        = base
        self.index_map   = index_map
        self.skip        = skip_unmapped
        if skip_unmapped:
            self._valid = [i for i in range(len(base))
                           if self._get_label(i) in index_map]
        else:
            self._valid = list(range(len(base)))

    def _get_label(self, i: int) -> int:
        item = self.base[i]
        return int(item[1])

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(self, idx: int):
        img, label = self.base[self._valid[idx]]
        return img, self.index_map.get(int(label), int(label))


def _subsample_per_class(dataset: Dataset, limit: int,
                         num_classes: int) -> Dataset:
    """Return a Subset with at most *limit* samples per class."""
    from torch.utils.data import Subset
    buckets: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for i in range(len(dataset)):          # type: ignore[arg-type]
        _, lbl = dataset[i]                 # type: ignore[index]
        cls = int(lbl)
        if cls in buckets and len(buckets[cls]) < limit:
            buckets[cls].append(i)
    indices = [i for lst in buckets.values() for i in lst]
    return Subset(dataset, indices)        # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Model / report loading
# ─────────────────────────────────────────────────────────────────────────────

def load_report(report_path: Path) -> Dict:
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(checkpoint_path: Path, report: Dict):
    from src.core.model import build_model
    model_name  = report["best_model_name"]
    num_classes = report["num_classes"]
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model = model.to(device).eval()
    print(f"[✓] Loaded {model_name}  ({num_classes} classes)  on {device}")
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# Test-set construction
# ─────────────────────────────────────────────────────────────────────────────

def build_test_loader(report: Dict,
                      sample_limit: Optional[int] = None,
                      batch_size: int = 64) -> DataLoader:
    """Reconstruct the test loader for any model described by *report*."""
    from src.core.transforms import get_transforms
    from src.training.incremental import (
        KNOWN_RAW_SOURCES, find_existing_source_root,
        canonicalize_source_label, normalize_label,
    )
    from torchvision.datasets import Food101, ImageFolder
    from torch.utils.data import ConcatDataset

    _, test_tf = get_transforms()
    class_names: List[str] = report["class_names"]
    class_to_idx = {n: i for i, n in enumerate(class_names)}
    num_classes   = len(class_names)

    parts: List[Dataset] = []

    # ── Food-101 test set ──────────────────────────────────────────────
    food101 = Food101(root=str(DATA_DIR), split="test",
                      download=False, transform=test_tf)
    f101_map = {
        food101.class_to_idx[cls]: class_to_idx[cls]
        for cls in food101.class_to_idx
        if cls in class_to_idx
    }
    parts.append(_RemappedDataset(food101, f101_map, skip_unmapped=True))
    print(f"[✓] Food-101 test  : {len(parts[-1]):>6,} images  "
          f"({len(f101_map)} classes mapped)")

    # ── Extra sources (incremental model only) ─────────────────────────
    training_mode  = report.get("training_mode", "baseline")
    used_sources   = {s["name"]
                      for s in report.get("data_summary", {})
                                     .get("extra_sources", [])}

    if training_mode == "incremental_finetune" and used_sources:
        for spec in KNOWN_RAW_SOURCES:
            if spec["name"] not in used_sources:
                continue
            source_root = find_existing_source_root(spec)
            if source_root is None:
                print(f"  [!] {spec['name']}: source root not found, skipping")
                continue
            test_subdir = spec.get("test_subdir")
            test_dir    = (source_root / test_subdir
                           if test_subdir and test_subdir != "."
                           else source_root)
            if not test_dir.is_dir():
                print(f"  [!] {spec['name']}: test dir missing ({test_dir})")
                continue
            try:
                folder = ImageFolder(str(test_dir), transform=test_tf)
            except Exception as exc:
                print(f"  [!] {spec['name']}: ImageFolder failed — {exc}")
                continue

            src_name  = spec["name"]
            remap: Dict[int, int] = {}
            for cls_name, folder_idx in folder.class_to_idx.items():
                canonical = canonicalize_source_label(src_name, cls_name)
                if canonical in class_to_idx:
                    remap[folder_idx] = class_to_idx[canonical]

            if not remap:
                print(f"  [!] {spec['name']}: no classes mapped, skipping")
                continue

            wrapped = _RemappedDataset(folder, remap, skip_unmapped=True)
            parts.append(wrapped)
            print(f"[✓] {spec['name']:40s}: {len(wrapped):>6,} images  "
                  f"({len(remap)} classes mapped)")

    dataset: Dataset = ConcatDataset(parts) if len(parts) > 1 else parts[0]  # type: ignore[arg-type]
    total = sum(len(p) for p in parts)       # type: ignore[arg-type]

    if sample_limit:
        dataset = _subsample_per_class(dataset, sample_limit, num_classes)
        print(f"[✓] Subsampled to ≤{sample_limit}/class → "
              f"{len(dataset):,} images")  # type: ignore[arg-type]
    else:
        print(f"[✓] Total test images: {total:,}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(loader: DataLoader, model: torch.nn.Module,
                  device: torch.device, num_classes: int
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred_top1, y_conf) arrays."""
    y_true, y_pred, y_conf = [], [], []
    with torch.inference_mode():
        for imgs, labels in tqdm(loader, desc="Evaluating", unit="batch"):
            imgs   = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs).float()
            probs  = torch.softmax(logits, dim=1)
            top1   = probs.argmax(dim=1)
            conf   = probs.max(dim=1).values
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(top1.cpu().tolist())
            y_conf.extend(conf.cpu().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_conf)


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_conf: np.ndarray,
                    class_names: List[str],
                    base_class_count: int) -> Dict:
    n = len(class_names)
    per_class = []
    for cls in range(n):
        mask    = y_true == cls
        total   = int(mask.sum())
        correct = int((y_pred[mask] == cls).sum()) if total else 0
        top1    = 100.0 * correct / total if total else float("nan")
        per_class.append({
            "class_name":  class_names[cls],
            "total":       total,
            "correct":     correct,
            "top1":        top1,
            "is_new":      cls >= base_class_count,
        })

    overall_top1 = 100.0 * (y_pred == y_true).mean()

    # Confusion matrix
    conf_mat = np.zeros((n, n), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        conf_mat[int(t), int(p)] += 1

    # Most-confused pairs  (off-diagonal, at least 1 actual sample)
    confused = []
    for i in range(n):
        for j in range(n):
            if i != j and conf_mat[i, j] > 0:
                confused.append((conf_mat[i, j], i, j))
    confused.sort(reverse=True)

    # Confidence split
    correct_mask  = y_pred == y_true
    correct_conf  = y_conf[correct_mask].tolist()
    wrong_conf    = y_conf[~correct_mask].tolist()

    # Per-class mean confidence
    cls_conf = []
    for cls in range(n):
        mask = y_true == cls
        cls_conf.append(float(y_conf[mask].mean()) if mask.sum() else float("nan"))

    return {
        "num_classes":    n,
        "base_classes":   base_class_count,
        "overall_top1":   round(overall_top1, 4),
        "total_images":   int(len(y_true)),
        "per_class":      per_class,
        "conf_mat":       conf_mat.tolist(),
        "top_confused":   [(int(c), int(i), int(j)) for c, i, j in confused[:50]],
        "correct_conf":   correct_conf,
        "wrong_conf":     wrong_conf,
        "cls_mean_conf":  cls_conf,
    }


def save_metrics(metrics: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[✓] Metrics saved → {path}")


def load_metrics(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_dark(fig, *axes):
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=DIM_TEXT)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.yaxis.grid(True, color=GRID_COL, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Per-class accuracy bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class(metrics: Dict, out_dir: Path) -> None:
    pc     = metrics["per_class"]
    n      = metrics["num_classes"]
    top1_o = metrics["overall_top1"]
    base_n = metrics["base_classes"]

    # Sort ascending by top1 (NaN classes at the right)
    pc_s = sorted(pc, key=lambda c: (np.isnan(c["top1"]), c["top1"]))
    names = [c["class_name"].replace("_", "\n") for c in pc_s]
    top1  = [c["top1"] if not np.isnan(c["top1"]) else 0 for c in pc_s]
    colors = [C_NEW if c["is_new"] else C_FOOD101 for c in pc_s]

    bar_w = 0.65
    x     = np.arange(n)
    fig_w = max(22, n * 0.30)
    fig, ax = plt.subplots(figsize=(fig_w, 9))
    _apply_dark(fig, ax)

    bars = ax.bar(x, top1, bar_w, color=colors, alpha=0.88, zorder=3)

    # Highlight very bad bars in red
    for bar, val in zip(bars, top1):
        if val < 50:
            bar.set_color(C_BAD)
            bar.set_alpha(0.92)

    ax.axhline(top1_o, color=C_FOOD101, linestyle="--", linewidth=1.4,
               alpha=0.65, label=f"Overall avg  {top1_o:.1f}%", zorder=2)
    ax.axhline(80, color=DIM_TEXT, linestyle=":", linewidth=0.9,
               alpha=0.5, zorder=2)
    ax.text(n - 0.3, 80.7, "80 % floor", color=DIM_TEXT,
            fontsize=7, va="bottom", ha="right")

    # Annotate worst 5
    for i, c in enumerate(pc_s[:5]):
        if np.isnan(c["top1"]):
            continue
        ax.annotate(f"{c['top1']:.0f}%",
                    xy=(i, c["top1"]),
                    xytext=(i, c["top1"] + 4),
                    fontsize=6.5, color=C_BAD, ha="center",
                    arrowprops=dict(arrowstyle="-", color="#e84c4c66", lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=5.5 if n > 130 else 7,
                        color=TEXT_COL, rotation=90, va="top")
    ax.set_yticks(range(0, 105, 10))
    ax.set_yticklabels([f"{v}%" for v in range(0, 105, 10)],
                        color=TEXT_COL, fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_title(
        f"Per-Class Top-1 Accuracy  ·  {n} classes  "
        f"(■ Food-101 baseline  ■ Incremental)",
        color="white", fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Class  (sorted by accuracy, ascending)",
                  color=DIM_TEXT, fontsize=10, labelpad=8)
    ax.set_ylabel("Top-1 Accuracy (%)", color=DIM_TEXT,
                  fontsize=10, labelpad=8)

    # Custom legend patches
    from matplotlib.patches import Patch
    legend_els = [
        Patch(color=C_FOOD101, alpha=0.88, label="Food-101 base"),
        Patch(color=C_NEW,     alpha=0.88, label="Incremental new"),
        Patch(color=C_BAD,     alpha=0.92, label="< 50 % accuracy"),
    ]
    leg = ax.legend(handles=legend_els, loc="lower right",
                    fontsize=9, framealpha=0.25,
                    labelcolor="white", facecolor="#222")
    leg.get_frame().set_edgecolor("#444")
    ax.add_artist(ax.legend(loc="upper left", fontsize=9, framealpha=0.25,
                            labelcolor="white", facecolor="#222"))

    # Colour patch for overall line legend
    ax.legend(handles=[
        plt.Line2D([0], [0], color=C_FOOD101, linestyle="--", lw=1.4,
                   label=f"Overall avg  {top1_o:.1f}%"),
        *legend_els,
    ], loc="lower right", fontsize=9, framealpha=0.25,
       labelcolor="white", facecolor="#222",
    ).get_frame().set_edgecolor("#444")

    plt.tight_layout(pad=1.5)
    _save(fig, out_dir / "eval_per_class.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Confusion analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion(metrics: Dict, out_dir: Path) -> None:
    class_names = [c["class_name"] for c in metrics["per_class"]]
    n           = metrics["num_classes"]
    conf_mat    = np.array(metrics["conf_mat"], dtype=np.float32)
    top_cnf     = metrics["top_confused"]   # [(count, true_idx, pred_idx), ...]

    # ── figure layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[1.1, 1.4, 0.55],
                            hspace=0.45, wspace=0.35)

    # ── (0,0-1)  Full matrix overview ─────────────────────────────────
    ax_full = fig.add_subplot(gs[0, :])
    ax_full.set_facecolor(PANEL_BG)
    for sp in ax_full.spines.values():
        sp.set_visible(False)

    # Normalise per row so rare/common classes are comparable
    row_sum = conf_mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    mat_norm = conf_mat / row_sum

    im_full = ax_full.imshow(mat_norm, aspect="auto", cmap="viridis",
                             interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(im_full, ax=ax_full, fraction=0.015, pad=0.01,
                 label="Row-normalised recall").ax.yaxis.set_tick_params(color=DIM_TEXT)
    ax_full.set_title(f"Full {n}×{n} Confusion Matrix  (row-normalised)",
                      color="white", fontsize=12, fontweight="bold")
    ax_full.set_xlabel("Predicted class index", color=DIM_TEXT, fontsize=9)
    ax_full.set_ylabel("True class index",      color=DIM_TEXT, fontsize=9)
    ax_full.tick_params(colors=DIM_TEXT, labelsize=7)

    # Diagonal reference line
    ax_full.plot([0, n - 1], [0, n - 1], color="#ffffff22", lw=0.7)

    # ── (1,0)  Zoomed: worst-K classes ────────────────────────────────
    worst_k  = min(30, n)
    pc_sort  = sorted(enumerate(metrics["per_class"]),
                      key=lambda x: (np.isnan(x[1]["top1"]), x[1]["top1"]))
    worst_idx = [idx for idx, _ in pc_sort[:worst_k]]
    sub_mat   = conf_mat[np.ix_(worst_idx, worst_idx)]
    sub_names = [class_names[i].replace("_", "\n") for i in worst_idx]

    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_zoom.set_facecolor(PANEL_BG)
    for sp in ax_zoom.spines.values():
        sp.set_visible(False)

    sub_norm = sub_mat / np.maximum(sub_mat.sum(axis=1, keepdims=True), 1)
    im_z = ax_zoom.imshow(sub_norm, aspect="auto", cmap="magma",
                          interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(im_z, ax=ax_zoom, fraction=0.046, pad=0.04)
    ax_zoom.set_xticks(range(worst_k))
    ax_zoom.set_yticks(range(worst_k))
    sz  = max(4, 7 - worst_k // 10)
    ax_zoom.set_xticklabels(sub_names, fontsize=sz, rotation=90,
                             color=TEXT_COL, va="top")
    ax_zoom.set_yticklabels(sub_names, fontsize=sz, color=TEXT_COL)
    ax_zoom.set_title(f"Zoomed: {worst_k} Worst Classes",
                      color="white", fontsize=11, fontweight="bold")
    ax_zoom.tick_params(colors=DIM_TEXT)

    # ── (1,1)  Top-20 most-confused pairs ─────────────────────────────
    ax_pairs = fig.add_subplot(gs[1, 1])
    ax_pairs.set_facecolor(PANEL_BG)
    ax_pairs.axis("off")

    rows = min(20, len(top_cnf))
    col_headers = ["#", "True class", "Predicted as", "Count"]
    table_data  = []
    for rank, (cnt, ti, pi) in enumerate(top_cnf[:rows], 1):
        table_data.append([
            str(rank),
            class_names[ti],
            class_names[pi],
            str(cnt),
        ])

    tbl = ax_pairs.table(
        cellText=table_data,
        colLabels=col_headers,
        cellLoc="left",
        loc="upper left",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(PANEL_BG if row % 2 == 0 else "#222222")
        cell.set_text_props(color=TEXT_COL if row > 0 else "white",
                            fontweight="bold" if row == 0 else "normal")
        cell.set_edgecolor(GRID_COL)
    ax_pairs.set_title("Top-20 Most-Confused Class Pairs",
                       color="white", fontsize=11, fontweight="bold", pad=10)

    # ── (2,:)  Class-level diagonal recall bar ─────────────────────────
    ax_diag = fig.add_subplot(gs[2, :])
    ax_diag.set_facecolor(PANEL_BG)
    for sp in ax_diag.spines.values():
        sp.set_visible(False)

    diag_recall = np.diag(mat_norm)   # recall per class
    sort_idx    = np.argsort(diag_recall)
    bar_clr     = [C_NEW if metrics["per_class"][i]["is_new"] else C_FOOD101
                   for i in sort_idx]
    ax_diag.bar(np.arange(n), diag_recall[sort_idx],
                color=bar_clr, alpha=0.85, width=1.0)
    ax_diag.axhline(diag_recall.mean(), color="white", linestyle="--",
                    linewidth=1.0, alpha=0.6,
                    label=f"Mean recall {diag_recall.mean():.3f}")
    ax_diag.set_xlim(-0.5, n - 0.5)
    ax_diag.set_ylim(0, 1.05)
    ax_diag.set_xlabel("Classes sorted by recall (ascending)",
                       color=DIM_TEXT, fontsize=9)
    ax_diag.set_ylabel("Recall", color=DIM_TEXT, fontsize=9)
    ax_diag.set_title("Per-Class Recall  (diagonal of confusion matrix)",
                      color="white", fontsize=11, fontweight="bold")
    ax_diag.tick_params(colors=DIM_TEXT)
    ax_diag.yaxis.grid(True, color=GRID_COL, linewidth=0.5)
    ax_diag.legend(fontsize=9, labelcolor="white",
                   facecolor="#222", framealpha=0.3,
                   ).get_frame().set_edgecolor("#444")

    _save(fig, out_dir / "eval_confusion.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Confidence analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_confidence(metrics: Dict, out_dir: Path) -> None:
    correct_c = np.array(metrics["correct_conf"])
    wrong_c   = np.array(metrics["wrong_conf"])
    cls_acc   = np.array([c["top1"] for c in metrics["per_class"]])
    cls_conf  = np.array(metrics["cls_mean_conf"])

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # ── (0,0)  Confidence histograms ──────────────────────────────────
    ax_hist = fig.add_subplot(gs[0, 0])
    _apply_dark(fig, ax_hist)
    bins = np.linspace(0, 1, 41)
    ax_hist.hist(correct_c, bins=bins, color=C_FOOD101, alpha=0.75,
                 label=f"Correct  (n={len(correct_c):,})", zorder=3)
    ax_hist.hist(wrong_c,   bins=bins, color=C_BAD,     alpha=0.75,
                 label=f"Wrong    (n={len(wrong_c):,})",   zorder=3)
    ax_hist.set_xlabel("Max softmax confidence", color=DIM_TEXT, fontsize=10)
    ax_hist.set_ylabel("Sample count",           color=DIM_TEXT, fontsize=10)
    ax_hist.set_title("Prediction Confidence Distribution",
                      color="white", fontsize=11, fontweight="bold")
    ax_hist.tick_params(colors=DIM_TEXT)
    leg = ax_hist.legend(fontsize=9, labelcolor="white",
                         facecolor="#222", framealpha=0.3)
    leg.get_frame().set_edgecolor("#444")

    # ── (0,1)  Per-class accuracy vs mean confidence scatter ──────────
    ax_scat = fig.add_subplot(gs[0, 1])
    _apply_dark(fig, ax_scat)
    valid = ~np.isnan(cls_acc) & ~np.isnan(cls_conf)
    colors_s = [C_NEW if c["is_new"] else C_FOOD101
                for c, v in zip(metrics["per_class"], valid) if v]
    ax_scat.scatter(cls_conf[valid], cls_acc[valid],
                    c=colors_s, s=22, alpha=0.75, zorder=3)
    # Ideal diagonal
    ax_scat.plot([0, 100], [0, 100], color="#ffffff33", lw=0.8, linestyle="--")
    ax_scat.set_xlim(0, 105)
    ax_scat.set_ylim(-2, 105)
    ax_scat.set_xlabel("Mean confidence per class (%)", color=DIM_TEXT, fontsize=10)
    ax_scat.set_ylabel("Top-1 accuracy per class (%)", color=DIM_TEXT, fontsize=10)
    ax_scat.set_title("Confidence vs Accuracy per Class",
                      color="white", fontsize=11, fontweight="bold")
    ax_scat.tick_params(colors=DIM_TEXT)
    from matplotlib.patches import Patch
    leg2 = ax_scat.legend(handles=[
        Patch(color=C_FOOD101, label="Food-101"),
        Patch(color=C_NEW,     label="Incremental"),
    ], fontsize=9, labelcolor="white", facecolor="#222", framealpha=0.3)
    leg2.get_frame().set_edgecolor("#444")

    # ── (1,0)  Calibration curve ──────────────────────────────────────
    ax_cal = fig.add_subplot(gs[1, 0])
    _apply_dark(fig, ax_cal)
    all_conf = np.concatenate([correct_c, wrong_c])
    all_corr = np.concatenate([np.ones(len(correct_c)),
                                np.zeros(len(wrong_c))])
    n_bins   = 15
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_cen = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (all_conf >= lo) & (all_conf < hi)
        if mask.sum() > 0:
            bin_acc.append(all_corr[mask].mean())
            bin_cen.append((lo + hi) / 2)
    ax_cal.plot([0, 1], [0, 1], color="#ffffff44",
                linestyle="--", linewidth=1.0, label="Perfect calibration")
    ax_cal.plot(bin_cen, bin_acc, "o-",
                color=C_TOP3, linewidth=1.8, markersize=5, label="Model")
    ax_cal.fill_between(bin_cen, bin_acc,
                         [b for b in bin_cen],
                         alpha=0.12, color=C_TOP3)
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1.05)
    ax_cal.set_xlabel("Mean predicted confidence", color=DIM_TEXT, fontsize=10)
    ax_cal.set_ylabel("Fraction correct",          color=DIM_TEXT, fontsize=10)
    ax_cal.set_title("Reliability / Calibration Curve",
                     color="white", fontsize=11, fontweight="bold")
    ax_cal.tick_params(colors=DIM_TEXT)
    leg3 = ax_cal.legend(fontsize=9, labelcolor="white",
                          facecolor="#222", framealpha=0.3)
    leg3.get_frame().set_edgecolor("#444")

    # ── (1,1)  Error rate vs confidence band ─────────────────────────
    ax_err = fig.add_subplot(gs[1, 1])
    _apply_dark(fig, ax_err)
    band_labels, err_rates, band_ns = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (all_conf >= lo) & (all_conf < hi)
        if mask.sum() > 0:
            err = 1.0 - all_corr[mask].mean()
            band_labels.append(f"{lo*100:.0f}–{hi*100:.0f}")
            err_rates.append(err * 100)
            band_ns.append(int(mask.sum()))
    xb = np.arange(len(band_labels))
    bar_colors = [C_BAD if e > 50 else C_TOP3 for e in err_rates]
    ax_err.bar(xb, err_rates, color=bar_colors, alpha=0.85, zorder=3)
    for xi, (er, ns) in enumerate(zip(err_rates, band_ns)):
        ax_err.text(xi, er + 0.8, f"{ns:,}", ha="center",
                    fontsize=6.5, color=DIM_TEXT)
    ax_err.set_xticks(xb)
    ax_err.set_xticklabels(band_labels, rotation=45, ha="right",
                            fontsize=7.5, color=TEXT_COL)
    ax_err.set_ylabel("Error rate (%)", color=DIM_TEXT, fontsize=10)
    ax_err.set_xlabel("Confidence band (%)", color=DIM_TEXT, fontsize=10)
    ax_err.set_title("Error Rate by Confidence Band",
                     color="white", fontsize=11, fontweight="bold")
    ax_err.tick_params(colors=DIM_TEXT)

    _save(fig, out_dir / "eval_confidence.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Group comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_groups(metrics: Dict, report: Dict, out_dir: Path) -> None:
    pc        = metrics["per_class"]
    base_n    = metrics["base_classes"]
    base_acc  = [c["top1"] for c in pc if not c["is_new"] and not np.isnan(c["top1"])]
    new_acc   = [c["top1"] for c in pc if     c["is_new"] and not np.isnan(c["top1"])]

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── (0,0)  Box plot ───────────────────────────────────────────────
    ax_box = fig.add_subplot(gs[0, 0])
    _apply_dark(fig, ax_box)
    data_grps  = [base_acc, new_acc]
    grp_labels = [f"Food-101\n(n={len(base_acc)})",
                  f"Incremental\n(n={len(new_acc)})"]
    bp = ax_box.boxplot(
        data_grps, patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=DIM_TEXT),
        capprops=dict(color=DIM_TEXT),
        flierprops=dict(marker="o", color=DIM_TEXT, alpha=0.5, markersize=4),
    )
    for patch, clr in zip(bp["boxes"], [C_FOOD101, C_NEW]):
        patch.set_facecolor(clr)
        patch.set_alpha(0.7)
    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels(grp_labels, color=TEXT_COL, fontsize=10)
    ax_box.set_ylabel("Top-1 accuracy (%)", color=DIM_TEXT, fontsize=10)
    ax_box.set_title("Accuracy Distribution by Class Group",
                     color="white", fontsize=11, fontweight="bold")
    ax_box.tick_params(colors=DIM_TEXT)

    # ── (0,1)  Histogram overlay ──────────────────────────────────────
    ax_hist2 = fig.add_subplot(gs[0, 1])
    _apply_dark(fig, ax_hist2)
    bins = np.linspace(0, 100, 21)
    ax_hist2.hist(base_acc, bins=bins, color=C_FOOD101, alpha=0.7,
                  label="Food-101", zorder=3)
    ax_hist2.hist(new_acc,  bins=bins, color=C_NEW,     alpha=0.7,
                  label="Incremental", zorder=3)
    ax_hist2.set_xlabel("Top-1 accuracy (%)",    color=DIM_TEXT, fontsize=10)
    ax_hist2.set_ylabel("Number of classes",     color=DIM_TEXT, fontsize=10)
    ax_hist2.set_title("Per-Class Accuracy Histogram by Group",
                       color="white", fontsize=11, fontweight="bold")
    ax_hist2.tick_params(colors=DIM_TEXT)
    leg = ax_hist2.legend(fontsize=9, labelcolor="white",
                           facecolor="#222", framealpha=0.3)
    leg.get_frame().set_edgecolor("#444")

    # ── (1,0)  Summary stats table ────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[1, 0])
    _apply_dark(fig, ax_tbl)
    ax_tbl.axis("off")

    def _stats(vals):
        a = np.array(vals)
        return {
            "mean":   f"{np.nanmean(a):.2f}%",
            "median": f"{np.nanmedian(a):.2f}%",
            "std":    f"{np.nanstd(a):.2f}%",
            "min":    f"{np.nanmin(a):.2f}%",
            "max":    f"{np.nanmax(a):.2f}%",
            "< 50%":  str(int((a < 50).sum())),
            "≥ 90%":  str(int((a >= 90).sum())),
        }

    s_base = _stats(base_acc)
    s_new  = _stats(new_acc)
    keys   = list(s_base.keys())
    tbl_data = [[k, s_base[k], s_new[k]] for k in keys]

    tbl = ax_tbl.table(
        cellText=tbl_data,
        colLabels=["Metric", "Food-101", "Incremental"],
        cellLoc="center",
        loc="center",
        bbox=[0.05, 0.0, 0.90, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(PANEL_BG if row % 2 == 0 else "#222222")
        cell.set_text_props(color="white" if row == 0 else TEXT_COL,
                            fontweight="bold" if row == 0 else "normal")
        cell.set_edgecolor(GRID_COL)
    ax_tbl.set_title("Statistical Summary",
                     color="white", fontsize=11, fontweight="bold", y=1.02)

    # ── (1,1)  Cumulative accuracy curve ─────────────────────────────
    ax_cdf = fig.add_subplot(gs[1, 1])
    _apply_dark(fig, ax_cdf)
    for vals, lbl, clr in [(base_acc, "Food-101", C_FOOD101),
                            (new_acc, "Incremental", C_NEW)]:
        a    = np.sort(vals)
        cdf  = np.arange(1, len(a) + 1) / len(a) * 100
        ax_cdf.plot(a, cdf, color=clr, linewidth=2.0, label=lbl)
        # 80-% accuracy threshold
        idx80 = np.searchsorted(a, 80)
        pct80 = cdf[idx80] if idx80 < len(cdf) else 100
        ax_cdf.axvline(80, color=clr, linestyle=":", linewidth=0.9, alpha=0.5)
        ax_cdf.text(80.5, pct80 - 3, f"{pct80:.0f}%\n≤80", fontsize=7,
                    color=clr, va="top")

    ax_cdf.axvline(80, color="#ffffff22", linestyle="--", linewidth=1.0)
    ax_cdf.set_xlim(0, 105)
    ax_cdf.set_ylim(0, 105)
    ax_cdf.set_xlabel("Top-1 accuracy threshold (%)", color=DIM_TEXT, fontsize=10)
    ax_cdf.set_ylabel("% of classes at or above threshold",
                      color=DIM_TEXT, fontsize=10)
    ax_cdf.set_title("Cumulative Class-Accuracy Curve",
                     color="white", fontsize=11, fontweight="bold")
    ax_cdf.tick_params(colors=DIM_TEXT)
    leg2 = ax_cdf.legend(fontsize=9, labelcolor="white",
                          facecolor="#222", framealpha=0.3)
    leg2.get_frame().set_edgecolor("#444")

    _save(fig, out_dir / "eval_groups.png")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(metrics: Dict, class_names: List[str]) -> None:
    pc     = metrics["per_class"]
    base_n = metrics["base_classes"]
    n      = metrics["num_classes"]

    base_top1 = [c["top1"] for c in pc if not c["is_new"] and not np.isnan(c["top1"])]
    new_top1  = [c["top1"] for c in pc if     c["is_new"] and not np.isnan(c["top1"])]

    print("\n" + "=" * 64)
    print("  MODEL EVALUATION SUMMARY")
    print("=" * 64)
    print(f"  Total classes      : {n}  "
          f"({base_n} Food-101 + {n - base_n} incremental)")
    print(f"  Total test images  : {metrics['total_images']:,}")
    print(f"  Overall Top-1      : {metrics['overall_top1']:.2f}%")
    print()
    if base_top1:
        print(f"  Food-101 classes   : mean {np.mean(base_top1):.1f}%  "
              f"median {np.median(base_top1):.1f}%")
    if new_top1:
        print(f"  Incremental classes: mean {np.mean(new_top1):.1f}%  "
              f"median {np.median(new_top1):.1f}%")
    print()
    worst5 = sorted(pc, key=lambda c: (np.isnan(c["top1"]), c["top1"]))[:5]
    print("  Worst 5 classes:")
    for c in worst5:
        tag = "[NEW]" if c["is_new"] else "     "
        acc = f"{c['top1']:.1f}%" if not np.isnan(c["top1"]) else "  n/a"
        print(f"    {tag}  {c['class_name']:35s}  {acc}")
    best5 = sorted(pc, key=lambda c: -(c["top1"] if not np.isnan(c["top1"]) else -1))[:5]
    print()
    print("  Best 5 classes:")
    for c in best5:
        tag = "[NEW]" if c["is_new"] else "     "
        print(f"    {tag}  {c['class_name']:35s}  {c['top1']:.1f}%")
    print("=" * 64 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive model evaluation with visual reports."
    )
    parser.add_argument(
        "--model", default=str(RUNS_DIR / "best_model.pth"),
        help="Path to model checkpoint (default: runs/best_model.pth)"
    )
    parser.add_argument(
        "--report", default=str(RUNS_DIR / "report.json"),
        help="Path to training report JSON (default: runs/report.json)"
    )
    parser.add_argument(
        "--cached", action="store_true",
        help="Skip inference; re-use outputs/eval_metrics.json from a prior run"
    )
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Max images per class (omit for full evaluation)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="B",
        help="Inference batch size (default: 64)"
    )
    parser.add_argument(
        "--out-dir", default=str(OUTPUTS_DIR),
        help="Directory for output plots and JSON (default: outputs/)"
    )
    args = parser.parse_args()

    out_dir     = Path(args.out_dir)
    report_path = Path(args.report)
    model_path  = Path(args.model)
    metrics_cache = out_dir / "eval_metrics.json"

    report = load_report(report_path)
    class_names  = report["class_names"]
    base_count   = report.get("data_summary", {}).get("base_class_count",
                                                       len(class_names))

    if args.cached:
        if not metrics_cache.exists():
            print(f"[!] Cached metrics not found at {metrics_cache}. "
                  "Run without --cached first.")
            sys.exit(1)
        print(f"[✓] Loading cached metrics from {metrics_cache}")
        metrics = load_metrics(metrics_cache)
    else:
        model, device = load_model(model_path, report)
        loader        = build_test_loader(report, args.sample, args.batch_size)
        y_true, y_pred, y_conf = run_inference(loader, model, device,
                                               report["num_classes"])
        metrics = compute_metrics(y_true, y_pred, y_conf,
                                  class_names, base_count)
        save_metrics(metrics, metrics_cache)

    print_summary(metrics, class_names)

    print("Generating plots …")
    plot_per_class(metrics, out_dir)
    plot_confusion(metrics, out_dir)
    plot_confidence(metrics, out_dir)
    plot_groups(metrics, report, out_dir)
    print(f"\nAll plots saved to  {out_dir}/")


if __name__ == "__main__":
    main()
