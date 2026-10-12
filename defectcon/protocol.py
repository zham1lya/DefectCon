"""Run configuration: fixed protocol, hyperparameters and training schedule.

The step budget is fixed per condition (21,780 steps without VF, 24,750 with
VF). The number of steps per epoch depends on the number of training normals
and on the batch composition, so the epoch count is derived per run.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from . import paths

HPARAM_KEYS = ("lr", "focal_gamma", "contrastive_weight", "temperature",
               "defects_per_batch")


def load_yaml(path: Path | str) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


def protocol() -> Dict[str, Any]:
    """The fixed protocol in ``configs/defectcon.yaml``."""
    return load_yaml(paths.CONFIGS_ROOT / "defectcon.yaml")


def load_hparams(name_or_path: str) -> Dict[str, Any]:
    """Hyperparameters by name (``configs/hparams/<name>.yaml``) or by path."""
    p = Path(name_or_path)
    if not p.suffix:
        p = paths.CONFIGS_ROOT / "hparams" / f"{name_or_path}.yaml"
    hp = load_yaml(p)
    missing = [k for k in HPARAM_KEYS if hp.get(k) is None]
    if missing:
        raise ValueError(f"{p}: missing hyperparameters {missing}")
    return {k: hp[k] for k in HPARAM_KEYS}


@dataclass
class Schedule:
    """Batch composition and length of one run."""
    batch_size: int
    defects_per_batch: int
    n_normals: int
    steps_per_epoch: int
    epochs: int
    warmup_epochs: int
    quotas: Dict[str, int]

    @property
    def total_steps(self) -> int:
        return self.epochs * self.steps_per_epoch


def make_schedule(*, n_normals: int, defects_per_batch: int, batch_size: int,
                  condition: str, budget_steps: Optional[int] = None) -> Schedule:
    """Derive the per-run schedule from the fixed step budget.

    One epoch is one pass over the training normals:
    ``steps_per_epoch = n_normals // (batch_size - defects_per_batch)``.
    Training runs the smallest whole number of epochs reaching the budget.
    Under VF the defect slots are split between real and synthetic images.
    """
    p = protocol()
    d = int(defects_per_batch)
    spe = n_normals // (batch_size - d)
    budget = (p["budget_blocks"][condition] * p["step_block"]
              if budget_steps is None else int(budget_steps))
    epochs = max(1, math.ceil(budget / spe))
    warmup = max(1, round(p["warmup_blocks"] * p["step_block"] / spe))
    if condition == "vf":
        real = round(p["real_defect_fraction_vf"] * d)
        quotas = {"normal": batch_size - d, "real_defect": real,
                  "synth_defect": d - real}
    else:
        quotas = {"normal": batch_size - d, "defect": d}
    return Schedule(batch_size, d, n_normals, spe, epochs, warmup, quotas)


@dataclass
class RunConfig:
    """Everything that defines one training run."""
    hparams: Dict[str, Any]
    condition: str = "novf"
    seed: int = 42
    batch_size: int = 128
    architecture: Dict[str, Any] = field(default_factory=dict)
    budget_steps: Optional[int] = None

    @classmethod
    def build(cls, hparams: Dict[str, Any], *, condition: str = "novf", seed: int = 42,
              ablation: Optional[str] = None, sensitivity: Optional[tuple] = None,
              budget_steps: Optional[int] = None) -> "RunConfig":
        """Base protocol + hyperparameters + one ablation or sensitivity change."""
        p = protocol()
        hp = copy.deepcopy(hparams)
        arch = dict(p["architecture"])
        batch_size = p["batch_size"]
        if ablation:
            cells = load_yaml(paths.CONFIGS_ROOT / "ablation.yaml")
            if ablation not in cells:
                raise ValueError(f"unknown ablation cell {ablation!r}: {list(cells)}")
            for key, value in (cells[ablation] or {}).items():
                (hp if key in HPARAM_KEYS else arch)[key] = value
        if sensitivity:
            axis, value = sensitivity
            if axis == "batch_size":
                # The defect fraction of the batch is held at its tuned value.
                frac = hp["defects_per_batch"] / batch_size
                batch_size = int(value)
                hp["defects_per_batch"] = max(1, min(batch_size - 1, round(frac * batch_size)))
            elif axis in HPARAM_KEYS:
                hp[axis] = int(value) if axis == "defects_per_batch" else float(value)
            else:
                raise ValueError(f"unknown sensitivity axis {axis!r}")
        return cls(hparams=hp, condition=condition, seed=seed, batch_size=batch_size,
                   architecture=arch, budget_steps=budget_steps)

    def schedule(self, n_normals: int) -> Schedule:
        return make_schedule(n_normals=n_normals,
                             defects_per_batch=self.hparams["defects_per_batch"],
                             batch_size=self.batch_size, condition=self.condition,
                             budget_steps=self.budget_steps)
