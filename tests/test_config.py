"""
Tests for src/training/config.py

Covers:
    Config.__post_init__ validation errors
    Config CPU-only auto-adjustment (batch_size, use_amp, num_workers)
    Config fast_mode behaviour
    Config models_to_train default
    IncrementalConfig.build_train_config
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# All Config construction is done inside functions so the patch context applies.


def _cpu_config(**kwargs):
    """Build a Config with CUDA mocked away so __post_init__ runs the CPU branch."""
    with patch("src.training.config.torch.cuda.is_available", return_value=False):
        from src.training.config import Config
        return Config(**kwargs)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_invalid_val_split_zero_raises(self):
        with pytest.raises(ValueError, match="val_split"):
            _cpu_config(val_split=0.0)

    def test_invalid_val_split_one_raises(self):
        with pytest.raises(ValueError, match="val_split"):
            _cpu_config(val_split=1.0)

    def test_invalid_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            _cpu_config(batch_size=0)

    def test_invalid_eval_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="eval_batch_size"):
            _cpu_config(eval_batch_size=0)

    def test_invalid_num_workers_negative_raises(self):
        with pytest.raises(ValueError, match="num_workers"):
            _cpu_config(num_workers=-1)

    def test_invalid_tta_num_views_zero_raises(self):
        with pytest.raises(ValueError, match="tta_num_views"):
            _cpu_config(tta_num_views=0)

    def test_valid_config_does_not_raise(self):
        cfg = _cpu_config()
        assert cfg is not None


# ---------------------------------------------------------------------------
# CPU branch auto-adjustment
# ---------------------------------------------------------------------------

class TestConfigCpuAdjustments:
    def test_use_amp_disabled_on_cpu(self):
        cfg = _cpu_config(use_amp=True)
        assert cfg.use_amp is False

    def test_large_batch_size_capped_on_cpu(self):
        cfg = _cpu_config(batch_size=64)
        assert cfg.batch_size <= 16

    def test_models_to_train_default_set(self):
        cfg = _cpu_config()
        assert isinstance(cfg.models_to_train, list)
        assert len(cfg.models_to_train) >= 1

    def test_explicit_models_to_train_preserved(self):
        cfg = _cpu_config(models_to_train=["resnet50"])
        assert cfg.models_to_train == ["resnet50"]


# ---------------------------------------------------------------------------
# fast_mode
# ---------------------------------------------------------------------------

class TestConfigFastMode:
    def test_fast_mode_disables_eda(self):
        cfg = _cpu_config(fast_mode=True)
        assert cfg.run_eda is False

    def test_fast_mode_reduces_models_to_one(self):
        cfg = _cpu_config(fast_mode=True, models_to_train=["resnet50", "efficientnet_b0"])
        assert len(cfg.models_to_train) == 1

    def test_fast_mode_with_single_model_keeps_it(self):
        cfg = _cpu_config(fast_mode=True, models_to_train=["resnet50"])
        assert cfg.models_to_train == ["resnet50"]


# ---------------------------------------------------------------------------
# IncrementalConfig
# ---------------------------------------------------------------------------

class TestIncrementalConfig:
    def test_build_train_config_returns_config(self):
        with patch("src.training.config.torch.cuda.is_available", return_value=False):
            from src.training.config import IncrementalConfig
            inc = IncrementalConfig()
            cfg = inc.build_train_config("resnet50")
            assert cfg.models_to_train == ["resnet50"]

    def test_build_train_config_inherits_lr(self):
        with patch("src.training.config.torch.cuda.is_available", return_value=False):
            from src.training.config import IncrementalConfig
            inc = IncrementalConfig(learning_rate=1e-5)
            cfg = inc.build_train_config("resnet50")
            assert cfg.learning_rate == 1e-5
