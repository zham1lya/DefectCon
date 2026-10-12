"""Computations behind the tables, figures and numbers of Section 5 and the appendices.

Every function reads the released files in ``results/`` only.
"""
from __future__ import annotations

import csv
from typing import Dict, List, Sequence

import numpy as np

from . import metrics, results, stats
from .results import TRAINING_SEEDS

METRIC_KEYS = ("ap", "pauc5", "pauc10", "auroc", "recall", "precision", "f1")
TPR_BUDGETS = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10)
PAUC_BOUNDS = np.round(np.arange(0.01, 0.1501, 0.005), 4)
ROC_GRID = np.linspace(0.0, 0.15, 1501)
HELDOUT_PREVALENCE = 150 / 5150
NATURAL_PREVALENCE = 0.0141          # dataset-level defect rate as reported (583 / 41,224)


# --------------------------------------------------------------------------
# absolute results
# --------------------------------------------------------------------------
def summary(method: str, condition: str, seeds: Sequence[int] = (42,)) -> Dict[str, tuple]:
    """Mean and SD over runs of each reported metric (Tables 3, 9, 10)."""
    table = results.collect(method, condition, seeds)
    keys = [k for k in METRIC_KEYS if k in next(iter(table.values()))]
    return {k: stats.mean_sd([m[k] for m in table.values()]) for k in keys}


def paired_vs(a: tuple, b: tuple, seeds: Sequence[int] = TRAINING_SEEDS) -> Dict[str, dict]:
    """Partition-level paired comparison ``a - b``; ``a``/``b`` = (method, condition)."""
    ta, tb = results.collect(*a, seeds), results.collect(*b, seeds)
    keys = [k for k in METRIC_KEYS if k in next(iter(ta.values())) and k in next(iter(tb.values()))]
    return {k: stats.paired(results.per_partition(ta, k), results.per_partition(tb, k)) for k in keys}


def variance(method: str, key: str) -> tuple:
    """(seed SD, partition SD) from the 10 partitions x 3 seeds of the main comparison."""
    t = results.collect(method, "main", TRAINING_SEEDS)
    grid = np.array([[t[(ps, k, s)][key] for s in TRAINING_SEEDS] for ps, k in results.partitions()])
    return stats.variance_components(grid)


def per_seed_ap() -> Dict[int, dict]:
    """AP per training seed: DefectCon mean, BGAD mean, paired difference."""
    out = {}
    for s in TRAINING_SEEDS:
        a = results.per_partition(results.collect("defectcon", "main", (s,)), "ap")
        b = results.per_partition(results.collect("bgad", "main", (s,)), "ap")
        out[s] = {"defectcon": float(a.mean()), "bgad": float(b.mean()), **stats.paired(a, b)}
    return out


def per_partition_ap() -> List[dict]:
    """Appendix E: held-out AP of every run of the main comparison."""
    dc = results.collect("defectcon", "main", TRAINING_SEEDS)
    bg = results.collect("bgad", "main", TRAINING_SEEDS)
    rows = []
    for ps, k in results.partitions():
        row = {"partition": f"{ps}/{k}"}
        for s in TRAINING_SEEDS:
            row[f"defectcon_s{s}"] = dc[(ps, k, s)]["ap"]
            row[f"bgad_s{s}"] = bg[(ps, k, s)]["ap"]
        row["delta"] = (np.mean([dc[(ps, k, s)]["ap"] for s in TRAINING_SEEDS])
                        - np.mean([bg[(ps, k, s)]["ap"] for s in TRAINING_SEEDS]))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# ensembles (Section 5.2.3)
# --------------------------------------------------------------------------
def ensembles(method: str, condition: str, seeds: Sequence[int] = TRAINING_SEEDS) -> dict:
    """Five-fold score-averaged ensembles, one per (partition seed, training seed)."""
    aps, aurocs, pauc5, rho = [], [], [], []
    for ps in results.PARTITION_SEEDS:
        for s in seeds:
            members = [results.load(method, condition, (ps, k, s)) for k in results.FOLDS]
            y = members[0][0]
            scores = [m[1] for m in members]
            e = stats.ensemble_scores(scores)
            aps.append(metrics.average_precision(y, e))
            aurocs.append(metrics.auroc(y, e))
            pauc5.append(metrics.pauc(y, e, 0.05))
            rho.append(stats.member_error_correlation(scores, y))
    return {"ap": stats.mean_sd(aps), "auroc": stats.mean_sd(aurocs),
            "pauc5": stats.mean_sd(pauc5), "error_correlation": float(np.mean(rho)), "n": len(aps)}


# --------------------------------------------------------------------------
# operating region (Section 5.3, Table 6, Fig. 3)
# --------------------------------------------------------------------------
def operating_region() -> dict:
    """TPR at fixed FPR, pAUC by upper bound, and mean ROC curves (seeds averaged per partition)."""
    per = {}
    for method in ("defectcon", "bgad"):
        tpr, pauc, roc = [], [], []
        for ps, k in results.partitions():
            t_, p_, r_ = [], [], []
            for s in TRAINING_SEEDS:
                y, sc = results.load(method, "main", (ps, k, s))
                t_.append(metrics.tpr_at_fpr(y, sc, TPR_BUDGETS))
                p_.append([metrics.pauc(y, sc, float(b)) for b in PAUC_BOUNDS])
                r_.append(metrics.tpr_at_fpr(y, sc, ROC_GRID))
            tpr.append(np.mean(t_, 0)); pauc.append(np.mean(p_, 0)); roc.append(np.mean(r_, 0))
        per[method] = {"tpr": np.array(tpr), "pauc": np.array(pauc), "roc": np.array(roc)}
    dc, bg = per["defectcon"], per["bgad"]
    out = {"tpr": {}, "pauc": {}}
    for j, b in enumerate(TPR_BUDGETS):
        out["tpr"][b] = {"defectcon": float(dc["tpr"][:, j].mean()), "bgad": float(bg["tpr"][:, j].mean()),
                         **stats.paired(dc["tpr"][:, j], bg["tpr"][:, j])}
    for j, b in enumerate(PAUC_BOUNDS):
        out["pauc"][float(b)] = {"defectcon": float(dc["pauc"][:, j].mean()),
                                 "bgad": float(bg["pauc"][:, j].mean()),
                                 **stats.paired(dc["pauc"][:, j], bg["pauc"][:, j])}
    d = dc["roc"].mean(0) - bg["roc"].mean(0)
    i = int(np.argmax(d < 0))
    out["roc_crossing"] = float(ROC_GRID[i - 1] + (ROC_GRID[i] - ROC_GRID[i - 1]) * (d[i - 1] / (d[i - 1] - d[i])))
    out["roc"] = {"grid": ROC_GRID, "defectcon_mean": dc["roc"].mean(0), "defectcon_sd": dc["roc"].std(0, ddof=1),
                  "bgad_mean": bg["roc"].mean(0), "bgad_sd": bg["roc"].std(0, ddof=1)}
    out["pauc5_mean_tpr_delta"] = float(np.mean([
        np.mean([metrics.pauc(*results.load("defectcon", "main", (ps, k, s)), 0.05, standardized=False)
                 - metrics.pauc(*results.load("bgad", "main", (ps, k, s)), 0.05, standardized=False)
                 for s in TRAINING_SEEDS]) for ps, k in results.partitions()]))
    return out


# --------------------------------------------------------------------------
# natural prevalence (Section 5.3)
# --------------------------------------------------------------------------
def precision_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    return tpr * prevalence / (tpr * prevalence + fpr * (1 - prevalence))


def natural_rate(region: dict | None = None) -> dict:
    """Precision and AP re-weighted from the held-out to the natural defect rate.

    Precision uses the mean TPR at each exact FPR (Table 6). AP is recomputed
    per run with every normal image weighted by the odds ratio of the two
    prevalences, then averaged over runs.
    """
    region = region or operating_region()
    pi = NATURAL_PREVALENCE
    w = (HELDOUT_PREVALENCE / (1 - HELDOUT_PREVALENCE)) / (pi / (1 - pi))
    out = {"enrichment": HELDOUT_PREVALENCE / pi, "precision": {}}
    for b, r in region["tpr"].items():
        out["precision"][b] = {m: precision_at_prevalence(r[m], b, pi) for m in ("defectcon", "bgad")}
    from sklearn.metrics import average_precision_score
    for method in ("defectcon", "bgad"):
        aps = []
        for run in results.runs(TRAINING_SEEDS):
            y, s = results.load(method, "main", run)
            aps.append(average_precision_score(y, s, sample_weight=np.where(y == 0, w, 1.0)))
        out[f"ap_{method}"] = float(np.mean(aps))
    return out


# --------------------------------------------------------------------------
# ablation (Section 5.4, Tables 7-8) and sensitivity (Appendix C)
# --------------------------------------------------------------------------
CELL_FILE_NAMES = {"D1": "D1", "P1": "P1", "J1": "proj", "L1": "L1"}


def ablation() -> Dict[str, dict]:
    """Absolute values and paired differences from A0 (seed-42 main runs), per cell."""
    a0 = results.collect("defectcon", "main", (42,))
    a0v = {k: np.array([a0[(ps, f, 42)][k] for ps, f in results.partitions()]) for k in ("ap", "auroc", "pauc5")}
    rows = {}
    for r in csv.DictReader(open(results.METRICS_ROOT / "ablation_cv_heldout.csv")):
        rows.setdefault(r["cell"], {})[r["partition"]] = r
    parts = [f"s{ps}f{k}" for ps, k in results.partitions()]
    out = {"A0": {k: stats.mean_sd(v) for k, v in a0v.items()}}
    for cell, name in CELL_FILE_NAMES.items():
        got = rows[name]
        vals = {k: np.array([float(got[p][c]) for p in parts])
                for k, c in (("ap", "ap"), ("auroc", "auroc"), ("pauc5", "pauc5_std"))}
        out[cell] = {k: stats.mean_sd(v) for k, v in vals.items()}
        out[cell]["paired"] = {k: stats.paired(vals[k], a0v[k]) for k in vals}
    return out


def sensitivity() -> List[dict]:
    """Table C rows: held-out AUROC, AP, pAUC5 of each sweep point (single runs, tuning split)."""
    rows = list(csv.DictReader(open(results.METRICS_ROOT / "sensitivity_heldout.csv")))
    l1 = next(r for r in csv.DictReader(open(results.METRICS_ROOT / "ablation_tune_heldout.csv")) if r["cell"] == "L1")
    out = [{"axis": r["param"], "point": r["value_tag"], "centre": r["source"] != "trained",
            "auroc": float(r["auroc"]), "ap": float(r["ap"]), "pauc5": float(r["pauc5_std"])} for r in rows]
    out.append({"axis": "contrastive_weight", "point": "lcon0", "centre": False,
                "auroc": float(l1["auroc"]), "ap": float(l1["ap"]), "pauc5": float(l1["pauc5_std"])})
    return out
