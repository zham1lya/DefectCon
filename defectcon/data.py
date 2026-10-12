"""Datasets and split loading."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset

from . import paths

SplitLists = Dict[str, List[str]]   # keys: normal, defect


def split_signature(*path_lists: Sequence[str]) -> str:
    """MD5 over the sorted image basenames; independent of the data location."""
    names = sorted(os.path.basename(p) for lst in path_lists for p in lst)
    return hashlib.md5("|".join(names).encode()).hexdigest()


def load_split(split_dir: Path) -> Tuple[SplitLists, SplitLists]:
    """Sorted training and validation lists of one split directory.

    Raises if a listed image is missing, so a run never trains on a silently
    reduced split.
    """
    out = []
    for part in ("train", "val"):
        lists = {}
        for kind in ("normal", "defect"):
            files = sorted(paths.read_split(split_dir / f"{part}_{kind}.txt"))
            missing = [f for f in files if not os.path.exists(f)]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} images of {split_dir}/{part}_{kind}.txt are missing "
                    f"(first: {missing[0]}). Set DEFECTCON_DATA.")
            lists[kind] = files
        out.append(lists)
    return out[0], out[1]


def load_grayscale(path: str) -> Image.Image:
    return Image.open(path).convert("L")


class TwoViewDataset(Dataset):
    """Two independently augmented views of each training image."""

    def __init__(self, normal: List[str], defect: List[str], transform: Callable):
        self.samples = [(p, 0) for p in normal] + [(p, 1) for p in defect]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = load_grayscale(path)
        return self.transform(img), self.transform(img), label


class EvalDataset(Dataset):
    """Single deterministic view per image, for validation and scoring."""

    def __init__(self, normal: List[str], defect: List[str], transform: Callable):
        self.samples = [(p, 0) for p in normal] + [(p, 1) for p in defect]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return self.transform(load_grayscale(path)), label
