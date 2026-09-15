"""Tests for Kaggle staging helpers (no network)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.training.datasets import _copy_tree, _stage_kaggle_extract, main as datasets_main


def test_copy_tree_roundtrip(tmp_path: Path):
    src = tmp_path / "cache" / "v1"
    src.mkdir(parents=True)
    (src / "a.txt").write_text("hi")
    sub = src / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("yo")

    dst_parent = tmp_path / "data" / "DS"
    n = _copy_tree(src, dst_parent, prefer_hardlink=False)
    assert n == 2
    staged = dst_parent / src.name
    assert (staged / "a.txt").read_text() == "hi"
    assert (staged / "sub" / "b.txt").read_text() == "yo"


def test_datasets_main_defaults_to_link_mode():
    with patch("src.training.datasets.download_and_stage") as mock_dl:
        datasets_main(full_copy=False)
        mock_dl.assert_called_once_with(fast_stage=True)


def test_datasets_main_full_copy_disables_link():
    with patch("src.training.datasets.download_and_stage") as mock_dl:
        datasets_main(full_copy=True)
        mock_dl.assert_called_once_with(fast_stage=False)


def test_stage_kaggle_extract_copy_mode(tmp_path: Path):
    cache = tmp_path / "hub" / "extract"
    cache.mkdir(parents=True)
    (cache / "f.jpg").write_bytes(b"\xff\xd8\xff")

    target_dir = tmp_path / "data" / "MySet"
    n = _stage_kaggle_extract(cache, target_dir, fast_stage=False)
    assert n == 1
    assert (target_dir / cache.name / "f.jpg").exists()
