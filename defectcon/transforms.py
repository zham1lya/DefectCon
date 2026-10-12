"""Image preprocessing.

Every DefectCon image is resized with the same bilinear ``resize`` at training,
validation and test time, then scaled to [0, 1] and normalised with mean 0.5
and standard deviation 0.5.
"""
from __future__ import annotations

import random

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .protocol import protocol


def resize(size: int) -> transforms.Resize:
    """The single resize used for DefectCon inputs (bilinear, square)."""
    return transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR)


def _normalize() -> transforms.Normalize:
    n = protocol()["normalize"]
    return transforms.Normalize(mean=[n["mean"]], std=[n["std"]])


class AddGaussianNoise:
    """Additive zero-mean Gaussian noise on a [0, 1] tensor, applied with probability ``p``."""

    def __init__(self, std: float = 0.03, p: float = 0.5):
        self.std = std
        self.p = p

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            t = t + torch.randn_like(t) * self.std
        return t


def eval_transform(size: int | None = None) -> transforms.Compose:
    """Deterministic preprocessing for validation, scoring and visualisation."""
    size = size or protocol()["image_size"]
    return transforms.Compose([resize(size), transforms.ToTensor(), _normalize()])


def train_transform(size: int | None = None) -> transforms.Compose:
    """Class-agnostic augmentation applied independently to each of the two views.

    No cropping or blurring, so that small defects are neither cut out nor
    smoothed away (Section 4.3.1).
    """
    size = size or protocol()["image_size"]
    return transforms.Compose([
        resize(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomAffine(
            degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02))], p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.7),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.03, p=0.5),
        _normalize(),
    ])
