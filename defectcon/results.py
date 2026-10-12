"""Access to the released per-image scores in ``results/scores``.

Layout: ``results/scores/<method>/<condition>/p<partition_seed>_f<fold>_s<seed>_<part>.npz``
with arrays ``path`` (image basename), ``label`` (1 = defect) and ``score``.

Methods: ``defectcon``, ``bgad``, ``supcon``, ``uninet``. Conditions: ``main`` (no VF, baseline
hyperparameters), ``vf`` (VF, baseline hyperparameters), ``vf_tuned`` (VF,
VF-tuned hyperparameters), ``vf_widened`` (VF, widened-search hyperparameters).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

from . import metrics, paths

PARTITION_SEEDS = (100, 101)
FOLDS = (0, 1, 2, 3, 4)
TRAINING_SEEDS = (42, 202, 303)
SCORES_ROOT = paths.RESULTS_ROOT / "scores"
METRICS_ROOT = paths.RESULTS_ROOT / "metrics"

Run = Tuple[int, int, int]   # (partition seed, fold, training seed)


def score_file(method: str, condition: str, run: Run, part: str, root: Path = SCORES_ROOT) -> Path:
    ps, k, s = run
    return root / method / condition / f"p{ps}_f{k}_s{s}_{part}.npz"


def load(method: str, condition: str, run: Run, part: str = "heldout",
         root: Path = SCORES_ROOT) -> Tuple[np.ndarray, np.ndarray]:
    """(labels, scores) of one run."""
    z = np.load(score_file(method, condition, run, part, root), allow_pickle=True)
    return z["label"], z["score"]


def has(method: str, condition: str, run: Run, part: str, root: Path = SCORES_ROOT) -> bool:
    return score_file(method, condition, run, part, root).exists()


def runs(seeds: Sequence[int] = (42,)) -> Iterator[Run]:
    """All runs in partition-major order: (100,0,s)...(101,4,s)."""
    for ps in PARTITION_SEEDS:
        for k in FOLDS:
            for s in seeds:
                yield ps, k, s


def partitions() -> Iterator[Tuple[int, int]]:
    for ps in PARTITION_SEEDS:
        for k in FOLDS:
            yield ps, k


def run_metrics(method: str, condition: str, run: Run, root: Path = SCORES_ROOT) -> Dict[str, float]:
    """The reported metrics of one run; operating-point metrics need validation scores."""
    y, s = load(method, condition, run, "heldout", root)
    if has(method, condition, run, "val", root):
        vy, vs = load(method, condition, run, "val", root)
        return metrics.summary_metrics(y, s, vy, vs)
    return metrics.summary_metrics(y, s)


def collect(method: str, condition: str, seeds: Sequence[int] = (42,),
            root: Path = SCORES_ROOT) -> Dict[Run, Dict[str, float]]:
    return {r: run_metrics(method, condition, r, root) for r in runs(seeds)}


def per_partition(table: Dict[Run, Dict[str, float]], key: str,
                  seeds: Optional[Sequence[int]] = None) -> np.ndarray:
    """Average the training seeds within each partition -> one value per partition."""
    seeds = seeds or sorted({r[2] for r in table})
    return np.array([np.mean([table[(ps, k, s)][key] for s in seeds]) for ps, k in partitions()])
