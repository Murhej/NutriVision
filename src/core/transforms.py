"""Standard ImageNet transforms for training and evaluation."""

from __future__ import annotations

from typing import Tuple

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(randaugment_magnitude: int = 7) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Return (train_transform, test_transform) with standard ImageNet normalization.

    Args:
        randaugment_magnitude: Magnitude for RandAugment (higher = stronger augmentation).
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=randaugment_magnitude),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.2)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, test_transform
