"""Statistics for the comparisons in Section 5.

Paired comparisons are made at the partition level: values of the training
seeds are averaged within each partition first, giving one difference per
partition (n = 10).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats
from scipy.stats import rankdata


def mean_sd(values: Sequence[float], ddof: int = 1) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=ddof)) if len(v) > ddof else float("nan")


def paired(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired comparison of ``a - b``.

    Returns the mean difference, its SD, a 95% Student-t interval, the same
    interval with the standard error doubled, the number of pairs with
    ``a > b``, two-sided paired t-test and Wilcoxon signed-rank p-values, and
    the standardised effect ``d_z = mean / SD``.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    sd = float(d.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    try:
        wp = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        wp = float("nan")
    return {"n": n, "mean": float(d.mean()), "sd": sd,
            "ci_lo": float(d.mean() - half), "ci_hi": float(d.mean() + half),
            "ci2_lo": float(d.mean() - 2 * half), "ci2_hi": float(d.mean() + 2 * half),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
            "t_p": float(stats.ttest_rel(a, b).pvalue), "wilcoxon_p": wp,
            "dz": float(d.mean() / sd) if sd > 0 else float("nan")}


def wilcoxon_floor(n: int) -> float:
    """Smallest two-sided Wilcoxon signed-rank p-value attainable with n pairs."""
    return float(stats.wilcoxon(np.arange(1.0, n + 1)).pvalue)


def variance_components(table: np.ndarray) -> tuple[float, float]:
    """One-way random-effects model ``y_ij = mu + a_i + e_ij``.

    ``table`` is partitions x seeds. Returns (seed SD, partition SD).
    """
    g = np.asarray(table, dtype=float)
    n_i, n_j = g.shape
    msw = ((g - g.mean(1, keepdims=True)) ** 2).sum() / (n_i * (n_j - 1))
    msb = n_j * ((g.mean(1) - g.mean()) ** 2).sum() / (n_i - 1)
    return float(np.sqrt(msw)), float(np.sqrt(max(msb - msw, 0.0) / n_j))


def ensemble_scores(member_scores: Sequence[np.ndarray]) -> np.ndarray:
    """Per-image mean of the member models' defect scores."""
    return np.vstack(member_scores).mean(axis=0)


def member_error_correlation(member_scores: Sequence[np.ndarray], y: np.ndarray) -> float:
    """Mean pairwise Pearson correlation of rank-based errors between members.

    Each member's scores are converted to normalised ranks and the label is
    subtracted, giving a per-image error; correlations are averaged over all
    member pairs.
    """
    z = np.stack([rankdata(s) / len(s) for s in member_scores]) - y[None, :]
    m = len(z)
    return float(np.mean([np.corrcoef(z[i], z[j])[0, 1]
                          for i in range(m) for j in range(i + 1, m)]))
