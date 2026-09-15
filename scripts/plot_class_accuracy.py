"""
Bar chart: Top-1 vs Top-3 accuracy for every Food-101 class.
Classes are sorted by Top-1 accuracy (ascending), so weak spots stand out on the left.

Usage:
    python scripts/plot_class_accuracy.py
    python scripts/plot_class_accuracy.py --output outputs/class_accuracy_bars.png
    python scripts/plot_class_accuracy.py --min-top1 0 --max-top1 80   # zoom in on worst classes
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = BASE_DIR / "outputs" / "per_class_performance.json"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "class_accuracy_bars.png"


def load_classes(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    classes = sorted(data["classes"], key=lambda c: c["top1_accuracy"])
    return data, classes


def plot(classes, overall_top1: float, overall_top3: float,
         output_path: Path, min_top1: float = 0.0, max_top1: float = 101.0):

    # Optional filter so you can zoom into the struggling tail
    classes = [c for c in classes if min_top1 <= c["top1_accuracy"] <= max_top1]
    if not classes:
        print("No classes match the filter range.")
        return

    n = len(classes)
    names   = [c["class_name"].replace("_", "\n") for c in classes]
    top1    = [c["top1_accuracy"] for c in classes]
    top3    = [c["top3_accuracy"] for c in classes]
    gap     = [t3 - t1 for t1, t3 in zip(top1, top3)]

    x = np.arange(n)
    bar_w = 0.42

    # Dynamic figure width: ~0.35 inches per class, min 18
    fig_w = max(18, n * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, 8))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a1a")

    bars_top1 = ax.bar(x - bar_w / 2, top1, bar_w,
                       label="Top-1", color="#4c9be8", alpha=0.92, zorder=3)
    bars_top3 = ax.bar(x + bar_w / 2, top3, bar_w,
                       label="Top-3", color="#f0a500", alpha=0.85, zorder=3)

    # Colour Top-1 bars red when gap > 20 pp — visually highlights "knows the neighbourhood but misses"
    for bar, g in zip(bars_top1, gap):
        if g > 20:
            bar.set_color("#e84c4c")
            bar.set_alpha(0.95)

    # Overall accuracy reference lines
    ax.axhline(overall_top1, color="#4c9be8", linestyle="--", linewidth=1.2,
               alpha=0.6, label=f"Avg Top-1 {overall_top1:.1f}%", zorder=2)
    ax.axhline(overall_top3, color="#f0a500", linestyle="--", linewidth=1.2,
               alpha=0.6, label=f"Avg Top-3 {overall_top3:.1f}%", zorder=2)

    # 80 % threshold line — production-quality floor
    ax.axhline(80, color="#888", linestyle=":", linewidth=1.0, alpha=0.5, zorder=2)
    ax.text(n - 0.3, 80.6, "80 % floor", color="#888", fontsize=7, va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.5, color="#cccccc", rotation=90, va="top")
    ax.set_yticks(range(0, 105, 10))
    ax.set_yticklabels([f"{v}%" for v in range(0, 105, 10)], color="#cccccc", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.8, n - 0.2)

    ax.set_title("Per-Class Top-1 vs Top-3 Accuracy  ·  Food-101",
                 color="white", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Food Class  (sorted by Top-1, ascending)", color="#aaaaaa", fontsize=10, labelpad=8)
    ax.set_ylabel("Accuracy (%)", color="#aaaaaa", fontsize=10, labelpad=8)

    ax.tick_params(axis="both", which="both", colors="#555", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(True, color="#333", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="lower right", fontsize=9, framealpha=0.25,
                       labelcolor="white", facecolor="#222")
    legend.get_frame().set_edgecolor("#444")

    # Annotate the worst 5 with their gap value
    for i, (c, g) in enumerate(zip(classes[:5], gap[:5])):
        ax.annotate(f"gap\n+{g:.0f}pp",
                    xy=(i - bar_w / 2, c["top1_accuracy"]),
                    xytext=(i, c["top3_accuracy"] + 3),
                    fontsize=6, color="#ff8080",
                    arrowprops=dict(arrowstyle="-", color="#ff808066", lw=0.8),
                    ha="center")

    plt.tight_layout(pad=1.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot per-class Top-1 vs Top-3 accuracy bar chart.")
    parser.add_argument("--input",     default=str(DEFAULT_INPUT),  help="Path to per_class_performance.json")
    parser.add_argument("--output",    default=str(DEFAULT_OUTPUT), help="Output PNG path")
    parser.add_argument("--min-top1",  type=float, default=0.0,     help="Only show classes with Top-1 >= this value")
    parser.add_argument("--max-top1",  type=float, default=101.0,   help="Only show classes with Top-1 <= this value")
    args = parser.parse_args()

    data, classes = load_classes(Path(args.input))
    print(f"Loaded {len(classes)} classes  |  "
          f"Overall Top-1: {data['overall_top1_accuracy']}%  "
          f"Top-3: {data['overall_top3_accuracy']}%")

    plot(
        classes,
        overall_top1=data["overall_top1_accuracy"],
        overall_top3=data["overall_top3_accuracy"],
        output_path=Path(args.output),
        min_top1=args.min_top1,
        max_top1=args.max_top1,
    )


if __name__ == "__main__":
    main()
