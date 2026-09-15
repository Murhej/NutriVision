"""
NutriVision — single CLI entrypoint.

Usage:
    python main.py                              # interactive menu
    python main.py train                        # train all configured models
    python main.py train --models resnet50      # train one model
    python main.py train --models resnet50,vit_b_16
    python main.py train --models all           # explicit all
    python main.py train --resume resnet50      # resume a model from its checkpoint
    python main.py train --resume all           # resume every model that has a checkpoint
    python main.py train --list                 # print available model names and exit
    python main.py incremental                  # incremental fine-tuning
    python main.py evaluate                     # per-class accuracy analysis on test set
    python main.py serve                        # start API server on :8000
    python main.py download                     # download & stage (default: link into kagglehub cache)
    python main.py download --full-copy         # copy files into data/ (slower; independent of cache)
"""

from __future__ import annotations

import argparse
import socket
import sys
from typing import List

from src.training.config import IncrementalConfig

# ---------------------------------------------------------------------------
# GPU banner
# ---------------------------------------------------------------------------

def _print_gpu_info() -> None:
    try:
        import torch
        print("\n" + "=" * 60)
        print("NutriVision")
        print("=" * 60)
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {name}  |  VRAM: {vram:.2f} GB")
        else:
            print("GPU: Not available -- running on CPU")
        print(f"PyTorch: {torch.__version__}")
        print("=" * 60 + "\n")
    except ImportError:
        print("PyTorch not found -- install requirements.txt first.")


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def _cmd_train(args: argparse.Namespace) -> None:
    from src.training.baseline import ALL_MODELS, main as train_main

    if getattr(args, "list", False):
        print("Available models:")
        for i, m in enumerate(ALL_MODELS, 1):
            print(f"  {i}. {m}")
        return

    # Resolve --models
    models_arg = getattr(args, "models", None)
    if models_arg and models_arg.lower() != "all":
        requested = [m.strip() for m in models_arg.split(",") if m.strip()]
        unknown = [m for m in requested if m not in ALL_MODELS]
        if unknown:
            print(f"Unknown model(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(ALL_MODELS)}")
            sys.exit(1)
        models_to_train = requested
    else:
        models_to_train = None  # use config defaults / all

    resume_model = getattr(args, "resume", None)
    train_main(models_to_train=models_to_train, resume_model=resume_model)


def _csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _cmd_incremental(args: argparse.Namespace) -> None:
    from src.training.incremental import main

    inc = IncrementalConfig()
    datasets_arg = getattr(args, "datasets", None)
    if datasets_arg:
        selected = _csv_list(datasets_arg)
        if not selected:
            raise ValueError("--datasets was provided but empty after parsing.")
        inc.selected_known_sources = selected

    if getattr(args, "only_datasets", False):
        inc.auto_discover_extra_data_dirs = False
        inc.use_raw_known_sources = True
        inc.extra_data_dirs = []

    base_checkpoint = getattr(args, "base_checkpoint", None)
    if base_checkpoint:
        inc.base_checkpoint_path = str(base_checkpoint)

    base_report = getattr(args, "base_report", None)
    if base_report:
        inc.base_report_path = str(base_report)

    output_checkpoint = getattr(args, "output_checkpoint", None)
    if output_checkpoint:
        inc.output_checkpoint_path = str(output_checkpoint)

    output_report = getattr(args, "output_report", None)
    if output_report:
        inc.output_report_path = str(output_report)

    if getattr(args, "no_promote", False):
        inc.promote_to_best = False

    replay_train = getattr(args, "replay_train", None)
    if replay_train is not None:
        inc.replay_train_samples = int(replay_train)

    replay_val = getattr(args, "replay_val", None)
    if replay_val is not None:
        inc.replay_val_samples = int(replay_val)

    replay_test = getattr(args, "replay_test", None)
    if replay_test is not None:
        inc.replay_test_samples = int(replay_test)

    main(inc)


def _cmd_evaluate(_args: argparse.Namespace) -> None:
    from src.evaluation.analyzer import analyze_per_class_performance
    analyze_per_class_performance()


def _cmd_serve(_args: argparse.Namespace) -> None:
    import uvicorn
    from src.api.app import app

    def _find_available_port(start_port: int = 8000, max_tries: int = 10) -> int:
        for port in range(start_port, start_port + max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(
            f"No available port found in range {start_port}-{start_port + max_tries - 1}."
        )

    port = _find_available_port(start_port=8000, max_tries=10)

    print("Starting NutriVision API Server...")
    print(f"  Docs:     http://localhost:{port}/docs")
    print(f"  Frontend: http://localhost:{port}/static/index.html")
    print("  Stop:     Ctrl+C\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def _cmd_download(args: argparse.Namespace) -> None:
    from src.training.datasets import main
    main(full_copy=bool(getattr(args, "full_copy", False)))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="NutriVision CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    # --- train ---
    train_p = sub.add_parser("train", help="Food-101 baseline training")
    train_p.add_argument(
        "--models", metavar="NAMES",
        help=(
            "Comma-separated model names to train, or 'all'. "
            "Example: --models resnet50,vit_b_16"
        ),
    )
    train_p.add_argument(
        "--resume", metavar="MODEL",
        help=(
            "Resume training from the last saved checkpoint. "
            "Pass a model name (e.g. resnet50) or 'all' to resume every model "
            "that has a checkpoint in runs/checkpoints/."
        ),
    )
    train_p.add_argument(
        "--list", action="store_true",
        help="Print available model names and exit.",
    )

    inc_p = sub.add_parser("incremental", help="Incremental fine-tuning on custom datasets")
    inc_p.add_argument(
        "--datasets",
        metavar="NAMES",
        help=(
            "Comma-separated known raw dataset names to include "
            "(e.g. fruits_360,vegfru,uec_food_256). "
            "Available names: uec_food_256, fruits_360, "
            "fruit_and_vegetable_image_recognition, vegfru, "
            "food_image_classification_dataset, indian_food_images_dataset, "
            "indonesian_food_dataset, fast_food_classification_dataset."
        ),
    )
    inc_p.add_argument(
        "--only-datasets",
        action="store_true",
        help=(
            "Strict mode for --datasets: disable custom extra_data_dirs and auto-discovery, "
            "so only the selected known raw datasets are used."
        ),
    )
    inc_p.add_argument(
        "--base-checkpoint",
        metavar="PATH",
        help="Checkpoint to start from (default: runs/best_model.pth).",
    )
    inc_p.add_argument(
        "--base-report",
        metavar="PATH",
        help="Report JSON to use as base metadata (default: runs/report.json).",
    )
    inc_p.add_argument(
        "--output-checkpoint",
        metavar="PATH",
        help="Where to save trained checkpoint if promoted (default: runs/best_model.pth).",
    )
    inc_p.add_argument(
        "--output-report",
        metavar="PATH",
        help="Where to save report JSON if promoted (default: runs/report.json).",
    )
    inc_p.add_argument(
        "--no-promote",
        action="store_true",
        help="Do not overwrite current best artifacts; write report only to last_incremental_report.json (or --output-report).",
    )
    inc_p.add_argument(
        "--replay-train",
        type=int,
        metavar="N",
        help="Replay sample count from Food-101 train split.",
    )
    inc_p.add_argument(
        "--replay-val",
        type=int,
        metavar="N",
        help="Replay sample count from Food-101 test split for validation.",
    )
    inc_p.add_argument(
        "--replay-test",
        type=int,
        metavar="N",
        help="Replay sample count from Food-101 test split for testing.",
    )
    sub.add_parser("evaluate",    help="Per-class accuracy analysis on the test set")
    sub.add_parser("serve",       help="Start the FastAPI server on port 8000")
    dl_p = sub.add_parser(
        "download",
        help=(
            "Download & stage Kaggle datasets into data/. "
            "Default: junction/symlink into the kagglehub cache (fast; data/ is bound to that cache)."
        ),
    )
    dl_p.add_argument(
        "--full-copy",
        action="store_true",
        help=(
            "Copy every file into data/ instead of linking to the kagglehub cache (slower, uses more disk). "
            "Use if you need a standalone data/ folder or plan to delete the cache. "
            "Or set NUTRIVISION_DOWNLOAD_FULL_COPY=1."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

COMMAND_FNS = {
    "train":       _cmd_train,
    "incremental": _cmd_incremental,
    "evaluate":    _cmd_evaluate,
    "serve":       _cmd_serve,
    "download":    _cmd_download,
}


def _interactive_menu() -> None:
    items = list(COMMAND_FNS.keys())
    descriptions = {
        "train":       "Food-101 baseline training (supports --models, --resume, --list)",
        "incremental": "Incremental fine-tuning on custom datasets",
        "evaluate":    "Per-class accuracy analysis on test set",
        "serve":       "Start the FastAPI server on port 8000",
        "download":    "Download & stage Kaggle datasets (default: link into kagglehub cache)",
    }
    print("Available commands:")
    for i, cmd in enumerate(items, 1):
        print(f"  [{i}] {cmd:<15} {descriptions[cmd]}")
    print("  [q] quit\n")

    choice = input("Select a command (number or name): ").strip().lower()
    if choice in ("q", "quit", "exit"):
        print("Bye!")
        return

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(items):
            choice = items[idx]
        else:
            print(f"Invalid selection.")
            return

    if choice in COMMAND_FNS:
        # For interactive menu, pass empty namespace (no extra flags)
        COMMAND_FNS[choice](argparse.Namespace())
    else:
        print(f"Unknown command: '{choice}'. Run `python main.py --help` for usage.")


def main() -> None:
    _print_gpu_info()

    parser = _build_parser()

    # No subcommand → interactive menu
    if len(sys.argv) < 2:
        _interactive_menu()
        return

    # Legacy: support plain `python main.py train` without subparser friction
    args = parser.parse_args()

    if args.command is None:
        _interactive_menu()
        return

    if args.command not in COMMAND_FNS:
        parser.print_help()
        sys.exit(1)

    COMMAND_FNS[args.command](args)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
