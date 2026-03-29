"""
Tests for main.py CLI argument parsing.

Strategy:
  - Test the argparse parser structure directly (no subprocess, no model loading).
  - Test _cmd_train's early-exit logic for unknown models and --list flag.
  - Keep everything isolated: mock train_main so no real training starts.
"""

from __future__ import annotations

import argparse
import sys
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Parser construction helper
# ---------------------------------------------------------------------------

def _build_parser():
    """Import and return the parser from main.py without triggering side effects."""
    import importlib
    import main as m
    # Build a fresh parser the same way main.py does
    return m._build_parser()


# ---------------------------------------------------------------------------
# Subcommand existence
# ---------------------------------------------------------------------------

def test_train_subcommand_exists():
    parser = _build_parser()
    args = parser.parse_args(["train"])
    assert args.command == "train"


def test_evaluate_subcommand_exists():
    parser = _build_parser()
    args = parser.parse_args(["evaluate"])
    assert args.command == "evaluate"


def test_serve_subcommand_exists():
    parser = _build_parser()
    args = parser.parse_args(["serve"])
    assert args.command == "serve"


def test_incremental_subcommand_exists():
    parser = _build_parser()
    args = parser.parse_args(["incremental"])
    assert args.command == "incremental"


# ---------------------------------------------------------------------------
# train --models flag
# ---------------------------------------------------------------------------

def test_train_models_single_model():
    parser = _build_parser()
    args = parser.parse_args(["train", "--models", "resnet50"])
    assert args.models == "resnet50"


def test_train_models_multiple_comma_separated():
    parser = _build_parser()
    args = parser.parse_args(["train", "--models", "resnet50,efficientnet_b0"])
    assert args.models == "resnet50,efficientnet_b0"


def test_train_models_all_keyword():
    parser = _build_parser()
    args = parser.parse_args(["train", "--models", "all"])
    assert args.models == "all"


def test_train_models_default_is_none_or_missing():
    parser = _build_parser()
    args = parser.parse_args(["train"])
    # --models is optional; value should be falsy when not provided
    assert not getattr(args, "models", None)


# ---------------------------------------------------------------------------
# train --resume flag
# ---------------------------------------------------------------------------

def test_train_resume_single_model():
    parser = _build_parser()
    args = parser.parse_args(["train", "--resume", "resnet50"])
    assert args.resume == "resnet50"


def test_train_resume_all():
    parser = _build_parser()
    args = parser.parse_args(["train", "--resume", "all"])
    assert args.resume == "all"


def test_train_resume_default_is_none():
    parser = _build_parser()
    args = parser.parse_args(["train"])
    assert getattr(args, "resume", None) is None


# ---------------------------------------------------------------------------
# train --list flag
# ---------------------------------------------------------------------------

def test_train_list_flag_set():
    parser = _build_parser()
    args = parser.parse_args(["train", "--list"])
    assert getattr(args, "list", False) is True


def test_train_list_default_false():
    parser = _build_parser()
    args = parser.parse_args(["train"])
    assert getattr(args, "list", False) is False


# ---------------------------------------------------------------------------
# _cmd_train logic: --list prints model names and returns
# ---------------------------------------------------------------------------

def test_cmd_train_list_prints_models_and_returns(capsys):
    from main import _cmd_train
    args = argparse.Namespace(list=True, models=None, resume=None)
    # Should NOT call train_main (no training)
    with patch("src.training.baseline.main") as mock_train:
        _cmd_train(args)
        mock_train.assert_not_called()

    captured = capsys.readouterr()
    assert "resnet50" in captured.out


# ---------------------------------------------------------------------------
# _cmd_train logic: unknown model exits with code 1
# ---------------------------------------------------------------------------

def test_cmd_train_unknown_model_exits_1():
    from main import _cmd_train
    args = argparse.Namespace(list=False, models="not_a_real_model_xyz", resume=None)

    with pytest.raises(SystemExit) as exc_info:
        _cmd_train(args)

    assert exc_info.value.code == 1


def test_cmd_train_unknown_model_prints_error(capsys):
    from main import _cmd_train
    args = argparse.Namespace(list=False, models="totally_fake", resume=None)

    with pytest.raises(SystemExit):
        _cmd_train(args)

    captured = capsys.readouterr()
    assert "totally_fake" in captured.out or "totally_fake" in captured.err


# ---------------------------------------------------------------------------
# _cmd_train logic: valid model delegates to train_main
# ---------------------------------------------------------------------------

def test_cmd_train_valid_model_calls_train_main():
    from main import _cmd_train
    args = argparse.Namespace(list=False, models="resnet50", resume=None)

    with patch("src.training.baseline.main") as mock_main:
        _cmd_train(args)
        mock_main.assert_called_once_with(
            models_to_train=["resnet50"],
            resume_model=None,
        )


def test_cmd_train_all_models_passes_none_to_train_main():
    from main import _cmd_train
    args = argparse.Namespace(list=False, models="all", resume=None)

    with patch("src.training.baseline.main") as mock_main:
        _cmd_train(args)
        mock_main.assert_called_once_with(
            models_to_train=None,
            resume_model=None,
        )


def test_cmd_train_no_models_arg_passes_none():
    from main import _cmd_train
    args = argparse.Namespace(list=False, models=None, resume=None)

    with patch("src.training.baseline.main") as mock_main:
        _cmd_train(args)
        mock_main.assert_called_once_with(
            models_to_train=None,
            resume_model=None,
        )
