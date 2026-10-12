"""Filesystem locations and split-list I/O.

Environment variables locate everything outside the repository:

    DEFECTCON_DATA     image root; contains noVF/{train,test}/{normal,defect}/
                       and VF/train/defect/
    DEFECTCON_SPLITS   split lists (heldout_*.txt, tune/, tune_vf/, cv_seed100/, ...)
    DEFECTCON_OUT      output root for training runs, checkpoints and scores
    DEFECTCON_RESULTS  results tree read by the analysis scripts (default: results/)

Split lists hold one image path per line.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_ROOT = REPO_ROOT / "configs"
FIGURES_ROOT = REPO_ROOT / "figures"
PAPER_NUMBERS = REPO_ROOT / "paper" / "paper_numbers.tex"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(os.path.abspath(os.path.expanduser(value))) if value else default


RESULTS_ROOT = _env_path("DEFECTCON_RESULTS", REPO_ROOT / "results")
DATA_ROOT = _env_path("DEFECTCON_DATA", REPO_ROOT / "data" / "images")
SPLITS_ROOT = _env_path("DEFECTCON_SPLITS", REPO_ROOT / "data" / "splits")
OUT_ROOT = _env_path("DEFECTCON_OUT", REPO_ROOT / "runs")

NOVF_ROOT = DATA_ROOT / "noVF"
TRAIN_NORMAL = NOVF_ROOT / "train" / "normal"
TRAIN_DEFECT = NOVF_ROOT / "train" / "defect"
TEST_NORMAL = NOVF_ROOT / "test" / "normal"
TEST_DEFECT = NOVF_ROOT / "test" / "defect"
VF_TRAIN_DEFECT = DATA_ROOT / "VF" / "train" / "defect"

HELDOUT_NORMAL_TXT = SPLITS_ROOT / "heldout_normal.txt"
HELDOUT_DEFECT_TXT = SPLITS_ROOT / "heldout_defective.txt"


def read_lines(path: os.PathLike | str) -> List[str]:
    """Non-empty, stripped lines of a text file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def read_split(path: os.PathLike | str) -> List[str]:
    """Image paths listed in a split file."""
    return read_lines(path)


def is_synthetic(path: str) -> bool:
    """True for a virtual-flaw image (stored under ``VF/`` of the data root)."""
    return (os.sep + "VF" + os.sep) in path


def resolve_heldout(names: List[str], kind: str) -> List[str]:
    """Resolve held-out basenames to image paths.

    The held-out lists store basenames; an image may sit in either the train or
    the test directory of its class. Raises if any image cannot be found.
    """
    search: Dict[str, List[Path]] = {"normal": [TRAIN_NORMAL, TEST_NORMAL],
                                     "defect": [TRAIN_DEFECT, TEST_DEFECT]}
    resolved, missing = [], []
    for name in names:
        name = os.path.basename(name)
        hit = next((d / name for d in search[kind] if (d / name).exists()), None)
        if hit is None:
            missing.append(name)
        else:
            resolved.append(str(hit))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(names)} held-out {kind} images not found under "
            f"{DATA_ROOT} (first: {missing[:3]}). Set DEFECTCON_DATA.")
    return resolved


def heldout_images() -> Dict[str, List[str]]:
    """The fixed held-out test set as ``{"normal": paths, "defect": paths}``."""
    return {"normal": resolve_heldout(read_lines(HELDOUT_NORMAL_TXT), "normal"),
            "defect": resolve_heldout(read_lines(HELDOUT_DEFECT_TXT), "defect")}


def split_dir(split: str, condition: str = "novf", partition_seed: int | None = None,
              fold: int | None = None) -> Path:
    """Directory holding the six ``{train,val,test}_{normal,defect}.txt`` lists.

    ``split="tune"`` is the tuning split; ``split="cv"`` is fold ``fold`` of the
    CV partition with seed ``partition_seed``.
    """
    suffix = "_vf" if condition == "vf" else ""
    if split == "tune":
        return SPLITS_ROOT / f"tune{suffix}"
    if split == "cv":
        if partition_seed is None or fold is None:
            raise ValueError("split='cv' needs partition_seed and fold")
        return SPLITS_ROOT / f"cv_seed{partition_seed}{suffix}" / f"fold{fold}"
    raise ValueError(f"unknown split {split!r}")
