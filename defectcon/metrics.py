"""Evaluation metrics (Section 5.1).

All metrics take labels ``y`` (1 = defect) and defect scores ``s``.

* AP: ``average_precision_score`` (step interpolation of the PR curve).
* pAUC: McClish-standardised partial AUC over ``FPR <= max_fpr``
  (``roc_auc_score(max_fpr=...)``); ``standardized=False`` returns the mean TPR
  over ``[0, max_fpr]``.
* Operating point: the threshold is the ``1 - fpr`` quantile of the model's
  own validation normals, applied unchanged to the test scores; an image is
  flagged when its score is strictly greater than the threshold.
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

OPERATING_FPR = 0.05


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def pauc(y: np.ndarray, s: np.ndarray, max_fpr: float, standardized: bool = True) -> float:
    """Partial AUC over ``FPR <= max_fpr``.

    ``standardized=True``: McClish standardisation (0.5 random, 1.0 perfect).
    ``standardized=False``: raw partial area divided by ``max_fpr``, i.e. the
    mean TPR over ``[0, max_fpr]`` with linear interpolation of the ROC curve.
    """
    if standardized:
        return float(roc_auc_score(y, s, max_fpr=max_fpr))
    fpr, tpr, _ = roc_curve(y, s)
    keep = fpr <= max_fpr
    f = np.append(fpr[keep], max_fpr)
    t = np.append(tpr[keep], np.interp(max_fpr, fpr, tpr))
    return float(np.trapz(t, f)) / max_fpr


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fprs: Iterable[float]) -> np.ndarray:
    """TPR read from the ROC curve at exact FPR values (linear interpolation)."""
    fpr, tpr, _ = roc_curve(y, s)
    return np.interp(np.asarray(list(fprs), dtype=float), fpr, tpr)


def calibrate_threshold(val_y: np.ndarray, val_s: np.ndarray, fpr: float = OPERATING_FPR) -> float:
    """Score threshold giving false-positive rate ``fpr`` on the validation normals."""
    return float(np.quantile(val_s[val_y == 0], 1.0 - fpr))


def operating_point(y: np.ndarray, s: np.ndarray, threshold: float) -> Dict[str, float]:
    """Recall, precision, F1 and realised FPR at a fixed threshold (``s > threshold``)."""
    pred = s > threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"recall": rec, "precision": prec, "f1": f1, "fpr": fp / max(1, int((y == 0).sum()))}


def summary_metrics(y: np.ndarray, s: np.ndarray, val_y: np.ndarray | None = None,
                    val_s: np.ndarray | None = None) -> Dict[str, float]:
    """The seven reported metrics; operating-point metrics need validation scores."""
    out = {"ap": average_precision(y, s), "pauc5": pauc(y, s, 0.05),
           "pauc10": pauc(y, s, 0.10), "auroc": auroc(y, s)}
    if val_y is not None:
        op = operating_point(y, s, calibrate_threshold(val_y, val_s))
        out.update(recall=op["recall"], precision=op["precision"], f1=op["f1"])
    return out
