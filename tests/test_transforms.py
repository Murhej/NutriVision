"""
Tests for src/core/transforms.py

Covers:
    get_transforms() — output tensor shape and value range
    train vs test transform differences
    randaugment_magnitude parameter effect on pipeline (doesn't crash)
    IMAGENET_MEAN / IMAGENET_STD constants
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from src.core.transforms import IMAGENET_MEAN, IMAGENET_STD, get_transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_pil(width: int = 300, height: int = 200) -> Image.Image:
    import numpy as np
    arr = (torch.rand(height, width, 3).numpy() * 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_imagenet_mean_has_three_channels():
    assert len(IMAGENET_MEAN) == 3


def test_imagenet_std_has_three_channels():
    assert len(IMAGENET_STD) == 3


def test_imagenet_mean_values_are_in_range():
    for v in IMAGENET_MEAN:
        assert 0.0 < v < 1.0


def test_imagenet_std_values_are_positive():
    for v in IMAGENET_STD:
        assert v > 0.0


# ---------------------------------------------------------------------------
# Test transform: shape and value range
# ---------------------------------------------------------------------------

def test_test_transform_output_shape(test_image_pil):
    _, test_tf = get_transforms()
    tensor = test_tf(test_image_pil)
    assert tensor.shape == (3, 224, 224)


def test_test_transform_output_is_float_tensor(test_image_pil):
    _, test_tf = get_transforms()
    tensor = test_tf(test_image_pil)
    assert tensor.dtype == torch.float32


def test_test_transform_values_in_plausible_range(test_image_pil):
    _, test_tf = get_transforms()
    tensor = test_tf(test_image_pil)
    # After ImageNet normalization, typical range is roughly [-3, 3]
    assert tensor.min() > -5.0
    assert tensor.max() <  5.0


def test_test_transform_larger_input_resized_correctly():
    img = _random_pil(640, 480)
    _, test_tf = get_transforms()
    tensor = test_tf(img)
    assert tensor.shape == (3, 224, 224)


def test_test_transform_small_input_resized_correctly():
    img = _random_pil(50, 50)
    _, test_tf = get_transforms()
    tensor = test_tf(img)
    assert tensor.shape == (3, 224, 224)


# ---------------------------------------------------------------------------
# Train transform: shape
# ---------------------------------------------------------------------------

def test_train_transform_output_shape(test_image_pil):
    train_tf, _ = get_transforms()
    tensor = train_tf(test_image_pil)
    assert tensor.shape == (3, 224, 224)


def test_train_transform_output_is_float_tensor(test_image_pil):
    train_tf, _ = get_transforms()
    tensor = train_tf(test_image_pil)
    assert tensor.dtype == torch.float32


# ---------------------------------------------------------------------------
# randaugment_magnitude parameter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("magnitude", [0, 5, 7, 10, 15])
def test_get_transforms_does_not_crash_for_magnitudes(magnitude, test_image_pil):
    train_tf, test_tf = get_transforms(randaugment_magnitude=magnitude)
    assert test_tf(test_image_pil).shape == (3, 224, 224)


# ---------------------------------------------------------------------------
# get_transforms returns exactly two transforms
# ---------------------------------------------------------------------------

def test_get_transforms_returns_tuple_of_two():
    result = get_transforms()
    assert len(result) == 2
