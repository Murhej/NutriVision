"""
Tests for checkpoint I/O, archive, and best-model promotion logic.

Covers:
    save_checkpoint      — epoch files + best.pth when is_best=True
    load_resume_checkpoint — happy path, each missing-file variant, malformed JSON
    _archive_model       — archive dir created, .pth + .json written with correct metadata
    _promote_best_so_far — new-best path, no-upgrade path, first-run path
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.training.trainer import load_resume_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state_dict() -> dict:
    """Minimal state dict: one tiny linear layer."""
    import torch.nn as nn
    m = nn.Linear(4, 2)
    return m.state_dict()


def _base_meta(global_epoch: int = 3) -> dict:
    return {
        "model_name": "resnet50",
        "global_epoch": global_epoch,
        "val_top1": 72.0,
        "val_top3": 88.0,
        "best_val_top1": 72.0,
        "best_val_top3": 88.0,
        "train_loss": 1.2,
    }


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------

class TestSaveCheckpoint:
    def test_creates_epoch_pth_and_json(self, tmp_path: Path):
        sd = _make_state_dict()
        meta = _base_meta(global_epoch=5)
        ckpt_dir = tmp_path / "checkpoints" / "resnet50"

        result = save_checkpoint(ckpt_dir, "resnet50", 5, sd, meta, is_best=False)

        assert result == ckpt_dir / "epoch_005.pth"
        assert (ckpt_dir / "epoch_005.pth").exists()
        assert (ckpt_dir / "epoch_005.json").exists()

    def test_json_contains_checkpoint_file_key(self, tmp_path: Path):
        sd = _make_state_dict()
        meta = _base_meta(global_epoch=2)
        ckpt_dir = tmp_path / "ck"

        save_checkpoint(ckpt_dir, "resnet50", 2, sd, meta, is_best=False)

        data = json.loads((ckpt_dir / "epoch_002.json").read_text())
        assert "checkpoint_file" in data
        assert data["global_epoch"] == 2

    def test_is_best_creates_best_files(self, tmp_path: Path):
        sd = _make_state_dict()
        meta = _base_meta(global_epoch=7)
        ckpt_dir = tmp_path / "ck"

        save_checkpoint(ckpt_dir, "resnet50", 7, sd, meta, is_best=True)

        assert (ckpt_dir / "best.pth").exists()
        assert (ckpt_dir / "best.json").exists()

    def test_not_best_does_not_create_best_files(self, tmp_path: Path):
        sd = _make_state_dict()
        ckpt_dir = tmp_path / "ck"

        save_checkpoint(ckpt_dir, "resnet50", 1, sd, _base_meta(1), is_best=False)

        assert not (ckpt_dir / "best.pth").exists()
        assert not (ckpt_dir / "best.json").exists()

    def test_creates_directory_if_missing(self, tmp_path: Path):
        ckpt_dir = tmp_path / "deep" / "nested" / "dir"
        assert not ckpt_dir.exists()

        save_checkpoint(ckpt_dir, "resnet50", 1, _make_state_dict(), _base_meta(1))

        assert ckpt_dir.exists()

    def test_saved_state_dict_is_loadable(self, tmp_path: Path):
        import torch.nn as nn
        sd = _make_state_dict()
        ckpt_dir = tmp_path / "ck"

        path = save_checkpoint(ckpt_dir, "resnet50", 1, sd, _base_meta(1), is_best=True)

        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert set(loaded.keys()) == set(sd.keys())


# ---------------------------------------------------------------------------
# load_resume_checkpoint
# ---------------------------------------------------------------------------

class TestLoadResumeCheckpoint:
    def test_returns_none_when_dir_missing(self, tmp_path: Path):
        ckpt_dir = tmp_path / "nonexistent"
        sd, meta = load_resume_checkpoint(ckpt_dir, "resnet50")
        assert sd is None
        assert meta is None

    def test_returns_none_when_only_best_pth_missing(self, tmp_path: Path):
        ckpt_dir = tmp_path / "ck"
        ckpt_dir.mkdir()
        (ckpt_dir / "best.json").write_text(json.dumps({"global_epoch": 3}))
        # best.pth deliberately absent

        sd, meta = load_resume_checkpoint(ckpt_dir, "resnet50")
        assert sd is None
        assert meta is None

    def test_returns_none_when_only_best_json_missing(self, tmp_path: Path):
        ckpt_dir = tmp_path / "ck"
        ckpt_dir.mkdir()
        torch.save(_make_state_dict(), ckpt_dir / "best.pth")
        # best.json deliberately absent

        sd, meta = load_resume_checkpoint(ckpt_dir, "resnet50")
        assert sd is None
        assert meta is None

    def test_returns_none_when_json_is_malformed(self, tmp_path: Path):
        ckpt_dir = tmp_path / "ck"
        ckpt_dir.mkdir()
        torch.save(_make_state_dict(), ckpt_dir / "best.pth")
        (ckpt_dir / "best.json").write_text("{ this is not valid json !!!}")

        sd, meta = load_resume_checkpoint(ckpt_dir, "resnet50")
        assert sd is None
        assert meta is None

    def test_happy_path_returns_state_dict_and_meta(self, tmp_path: Path):
        ckpt_dir = tmp_path / "ck"
        original_sd = _make_state_dict()
        original_meta = _base_meta(global_epoch=4)

        save_checkpoint(ckpt_dir, "resnet50", 4, original_sd, original_meta, is_best=True)
        sd, meta = load_resume_checkpoint(ckpt_dir, "resnet50")

        assert sd is not None
        assert meta is not None
        assert set(sd.keys()) == set(original_sd.keys())
        assert meta["global_epoch"] == 4
        assert meta["val_top1"] == 72.0

    def test_returns_none_when_pth_is_corrupted(self, tmp_path: Path):
        ckpt_dir = tmp_path / "ck"
        ckpt_dir.mkdir()
        (ckpt_dir / "best.pth").write_bytes(b"not a valid pytorch file")
        (ckpt_dir / "best.json").write_text(json.dumps({"global_epoch": 1}))

        sd, meta = load_resume_checkpoint(ckpt_dir, "resnet50")
        assert sd is None
        assert meta is None


# ---------------------------------------------------------------------------
# _archive_model
# ---------------------------------------------------------------------------

class TestArchiveModel:
    def test_creates_archive_dir_and_files(self, tmp_path: Path):
        from src.training.baseline import _archive_model

        weights = tmp_path / "model.pth"
        torch.save(_make_state_dict(), weights)
        runs_dir = tmp_path / "runs"

        result = _archive_model(str(weights), "resnet50", {"test_top1_accuracy": 85.0}, str(runs_dir))

        archive_dir = runs_dir / "archive"
        assert archive_dir.exists()
        archived_pth = Path(result)
        assert archived_pth.exists()
        assert archived_pth.suffix == ".pth"
        assert "resnet50" in archived_pth.name

    def test_creates_matching_json_metadata(self, tmp_path: Path):
        from src.training.baseline import _archive_model

        weights = tmp_path / "model.pth"
        torch.save(_make_state_dict(), weights)
        runs_dir = tmp_path / "runs"
        metrics = {"test_top1_accuracy": 85.0, "test_top3_accuracy": 94.0}

        result = _archive_model(str(weights), "resnet50", metrics, str(runs_dir))

        json_path = Path(result).with_suffix(".json")
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["model_name"] == "resnet50"
        assert data["test_top1_accuracy"] == 85.0

    def test_multiple_archives_do_not_overwrite(self, tmp_path: Path):
        from src.training.baseline import _archive_model
        import time as _time

        weights = tmp_path / "model.pth"
        torch.save(_make_state_dict(), weights)
        runs_dir = tmp_path / "runs"

        path1 = _archive_model(str(weights), "resnet50", {}, str(runs_dir))
        _time.sleep(1.1)  # ensure different timestamp
        path2 = _archive_model(str(weights), "resnet50", {}, str(runs_dir))

        assert path1 != path2
        assert Path(path1).exists()
        assert Path(path2).exists()


# ---------------------------------------------------------------------------
# _promote_best_so_far
# ---------------------------------------------------------------------------

class TestPromoteBestSoFar:
    """
    Unit tests for the cross-run best-model promotion logic.
    We supply fully-formed `results` dicts and check that best_model.pth
    and report.json are written / not-written correctly.
    """

    def _make_config(self):
        from unittest.mock import patch as _patch
        with _patch("torch.cuda.is_available", return_value=False):
            from src.training.config import Config
            return Config(models_to_train=["resnet50"])

    def _make_results(self, runs_dir: Path, top1: float, top3: float) -> dict:
        """Create a minimal results dict with a real weights file."""
        weights = runs_dir / "resnet50_weights.pth"
        torch.save(_make_state_dict(), weights)
        return {
            "resnet50": {
                "best_val_top1": top1,
                "best_val_top3": top3,
                "final_top1": top1,
                "final_top3": top3,
                "weights_path": str(weights),
            }
        }

    def test_empty_results_returns_without_writing(self, tmp_path: Path):
        from src.training.baseline import _promote_best_so_far
        config = self._make_config()

        _promote_best_so_far({}, [], [], config, str(tmp_path), time.time(), ["resnet50"])

        assert not (tmp_path / "best_model.pth").exists()
        assert not (tmp_path / "report.json").exists()

    def test_first_run_writes_best_model_and_report(self, tmp_path: Path):
        from src.training.baseline import _promote_best_so_far
        config = self._make_config()
        results = self._make_results(tmp_path, top1=85.0, top3=94.0)

        _promote_best_so_far(
            results, [{"model": "resnet50"}], ["apple_pie"], config,
            str(tmp_path), time.time(), ["resnet50"],
        )

        assert (tmp_path / "best_model.pth").exists()
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["best_model_name"] == "resnet50"
        assert report["best_model_metrics"]["val_top3_accuracy"] == 94.0

    def test_new_model_beats_existing_best(self, tmp_path: Path):
        from src.training.baseline import _promote_best_so_far
        config = self._make_config()

        # Existing deployed best at 88% top3
        existing_report = {
            "best_model_name": "efficientnet_b0",
            "best_model_metrics": {
                "val_top1_accuracy": 80.0,
                "val_top3_accuracy": 88.0,
                "test_top1_accuracy": 80.0,
                "test_top3_accuracy": 88.0,
            },
        }
        (tmp_path / "report.json").write_text(json.dumps(existing_report))

        # New model at 92% top3 — should win
        results = self._make_results(tmp_path, top1=85.0, top3=92.0)
        _promote_best_so_far(
            results, [], ["apple_pie"], config,
            str(tmp_path), time.time(), ["resnet50"],
        )

        report = json.loads((tmp_path / "report.json").read_text())
        assert report["best_model_name"] == "resnet50"
        assert report["best_model_metrics"]["val_top3_accuracy"] == 92.0

    def test_no_upgrade_when_new_model_is_worse(self, tmp_path: Path):
        from src.training.baseline import _promote_best_so_far
        config = self._make_config()

        # Existing deployed best at 95% top3
        existing_best_pth = tmp_path / "best_model.pth"
        torch.save(_make_state_dict(), existing_best_pth)
        existing_report = {
            "best_model_name": "vit_b_16",
            "best_model_metrics": {
                "val_top1_accuracy": 89.0,
                "val_top3_accuracy": 95.0,
                "test_top1_accuracy": 89.0,
                "test_top3_accuracy": 95.0,
            },
        }
        (tmp_path / "report.json").write_text(json.dumps(existing_report))
        original_mtime = existing_best_pth.stat().st_mtime

        # New model only at 85% top3 — should NOT displace vit_b_16
        results = self._make_results(tmp_path, top1=78.0, top3=85.0)
        _promote_best_so_far(
            results, [], ["apple_pie"], config,
            str(tmp_path), time.time(), ["resnet50"],
        )

        # best_model.pth should NOT have changed
        assert existing_best_pth.stat().st_mtime == original_mtime

        report = json.loads((tmp_path / "report.json").read_text())
        assert report["best_model_name"] == "vit_b_16"
        assert report["best_model_metrics"]["val_top3_accuracy"] == 95.0

    def test_report_status_in_progress_when_partial(self, tmp_path: Path):
        from src.training.baseline import _promote_best_so_far
        config = self._make_config()
        results = self._make_results(tmp_path, top1=80.0, top3=91.0)

        _promote_best_so_far(
            results, [], ["c1"], config,
            str(tmp_path), time.time(),
            models_requested=["resnet50", "vit_b_16"],  # 2 requested, 1 done
        )

        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "in_progress"

    def test_report_status_complete_when_all_done(self, tmp_path: Path):
        from src.training.baseline import _promote_best_so_far
        config = self._make_config()
        results = self._make_results(tmp_path, top1=80.0, top3=91.0)

        _promote_best_so_far(
            results, [], ["c1"], config,
            str(tmp_path), time.time(),
            models_requested=["resnet50"],  # exactly 1 requested, 1 done
        )

        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "complete"
