"""Checkpoint loading, per-image scoring and embedding extraction.

The defect score of an image is ``softmax(logits)[1]`` of the final model,
computed on the bilinear ``eval_transform`` input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_grayscale
from .model import DefectCon
from .protocol import protocol
from .transforms import eval_transform


def checkpoint_arch(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Architecture of a checkpoint.

    Checkpoints written by ``scripts/train.py`` and the ablation/sensitivity
    checkpoints carry an ``arch`` entry; the CV checkpoints (``*_last.pth``)
    do not and use the fixed architecture of ``configs/defectcon.yaml``.
    """
    base = dict(protocol()["architecture"])
    arch = ckpt.get("arch")
    if not arch:
        return base
    if arch.get("attention_type", "none") != "none" or arch.get("cls_tap", "pre") != "pre":
        raise ValueError(f"unsupported architecture in checkpoint: {arch}")
    return {"dilate": arch.get("dilate") or "none",
            "pooling": arch.get("pooling") or "avg",
            "proj_type": arch.get("proj_type") or "mlp",
            "proj_dim": arch.get("proj_dim") or base["proj_dim"],
            "hidden_dim": arch.get("hidden_dim") or base["hidden_dim"]}


def load_checkpoint(path: Path | str, device: torch.device) -> Tuple[DefectCon, Dict[str, Any]]:
    """Rebuild the model of a checkpoint and load its weights (strict)."""
    from .engine import deterministic_cudnn
    deterministic_cudnn()
    ckpt = torch.load(path, map_location=device)
    arch = checkpoint_arch(ckpt)
    model = DefectCon.from_arch(arch)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def _batches(paths: List[str], size: int):
    transform = eval_transform()
    for i in range(0, len(paths), size):
        yield torch.stack([transform(load_grayscale(p)) for p in paths[i:i + size]])


@torch.no_grad()
def score_images(model: DefectCon, normal: List[str], defect: List[str],
                 device: torch.device, batch_size: int | None = None) -> Dict[str, np.ndarray]:
    """Defect scores for ``normal + defect`` images, in that order."""
    batch_size = batch_size or protocol()["score_batch_size"]
    model.eval()
    images = list(normal) + list(defect)
    scores = []
    for x in _batches(images, batch_size):
        scores.append(F.softmax(model(x.to(device)).float(), dim=1)[:, 1].cpu().numpy())
    return {"path": np.array(images),
            "label": np.array([0] * len(normal) + [1] * len(defect)),
            "score": np.concatenate(scores) if scores else np.zeros(0)}


@torch.no_grad()
def embed_images(model: DefectCon, images: List[str], device: torch.device,
                 batch_size: int | None = None) -> np.ndarray:
    """L2-normalised 512-d encoder embeddings of ``images``."""
    batch_size = batch_size or protocol()["score_batch_size"]
    model.eval()
    return np.concatenate([model.embed(x.to(device)).float().cpu().numpy()
                           for x in _batches(images, batch_size)])


def score_split(model: DefectCon, split_dir: Path, device: torch.device) -> Dict[str, Dict[str, np.ndarray]]:
    """Scores on a split's validation lists and on the held-out test set.

    Image paths are stored as basenames.
    """
    from . import paths
    val = {k: paths.read_split(Path(split_dir) / f"val_{k}.txt") for k in ("normal", "defect")}
    held = paths.heldout_images()
    out = {}
    for name, lists in (("val", val), ("heldout", held)):
        s = score_images(model, lists["normal"], lists["defect"], device)
        s["path"] = np.array([Path(p).name for p in s["path"]])
        out[name] = s
    return out


def save_scores(path: Path, scores: Dict[str, np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, path=scores["path"], label=scores["label"], score=scores["score"])
