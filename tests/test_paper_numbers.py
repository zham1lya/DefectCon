"""Every number macro of the paper text (paper/paper_numbers.tex), recomputed from results/.

Values are compared at the precision the paper prints, allowing one unit in the
last printed digit (values are rounded from 4-decimal intermediates, so a
second rounding can move the last digit by one). Win counts must match exactly.
Macros that cannot be recomputed from the released files are listed in
NOT_RECOMPUTED with the reason.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from typing import Dict, Union

import numpy as np
import pytest

from defectcon import analysis, paths, results, stats

Value = Union[float, str, tuple]

NOT_RECOMPUTED: Dict[str, str] = {}

_MACRO = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}")


def paper_macros() -> Dict[str, str]:
    text = paths.PAPER_NUMBERS.read_text(encoding="utf-8")
    return {m.group(1): m.group(2).strip() for m in _MACRO.finditer(text)}


def wins(n: int, total: int = 10) -> str:
    return f"{n}/{total}"


@lru_cache(maxsize=1)
def computed() -> Dict[str, Value]:
    c: Dict[str, Value] = {}
    key = {"AP": "ap", "Pfive": "pauc5", "Pten": "pauc10", "AUROC": "auroc",
           "Rec": "recall", "Prec": "precision", "Fone": "f1"}

    # Table 3: absolute results, 30 runs per method
    for prefix, method in (("dc", "defectcon"), ("bg", "bgad")):
        s = analysis.summary(method, "main", results.TRAINING_SEEDS)
        for name, k in key.items():
            c[f"{prefix}{name}"], c[f"{prefix}{name}sd"] = s[k]

    # Table 4: paired at partition level
    p = analysis.paired_vs(("defectcon", "main"), ("bgad", "main"))
    for name, k in key.items():
        r = p[k]
        c[f"d{name}"], c[f"d{name}sd"] = r["mean"], r["sd"]
        c[f"d{name}lo"], c[f"d{name}hi"] = r["ci_lo"], r["ci_hi"]
        c[f"d{name}wins"] = wins(r["wins"])
        c[f"d{name}tp"], c[f"d{name}wp"], c[f"d{name}dz"] = r["t_p"], r["wilcoxon_p"], r["dz"]
    c["dPrecr"], c["dFoner"] = p["precision"]["mean"], p["f1"]["mean"]
    c["dblAP"] = (p["ap"]["ci2_lo"], p["ap"]["ci2_hi"])
    c["dblP"] = (p["pauc5"]["ci2_lo"], p["pauc5"]["ci2_hi"])
    c["wfloor"] = stats.wilcoxon_floor(10)

    # Table 5: variance components
    for prefix, method in (("DC", "defectcon"), ("BG", "bgad")):
        for name in ("AP", "Pfive", "AUROC"):
            c[f"vc{prefix}seed{name}"], c[f"vc{prefix}part{name}"] = analysis.variance(method, key[name])
    c["vcDCseedPrecr"] = analysis.variance("defectcon", "precision")[0]
    c["vcDCseedFoner"] = analysis.variance("defectcon", "f1")[0]

    # Section 5.2.2: per training seed
    for suffix, (seed, r) in zip("abc", analysis.per_seed_ap().items()):
        c[f"psDC{suffix}"], c[f"psBG{suffix}"] = r["defectcon"], r["bgad"]
        c[f"psD{suffix}r"] = r["mean"]

    # Section 5.2.3: ensembles
    e = {m: analysis.ensembles(m, "main") for m in ("defectcon", "bgad")}
    for prefix, m in (("DC", "defectcon"), ("BG", "bgad")):
        c[f"ens{prefix}AP"], c[f"ens{prefix}APsd"] = e[m]["ap"]
        c[f"ens{prefix}AUROC"] = e[m]["auroc"][0]
        c[f"ensRho{prefix}"] = e[m]["error_correlation"]

    # Table 6 and Section 5.3
    region = analysis.operating_region()
    for suffix, b in zip("abcegj", analysis.TPR_BUDGETS):
        r = region["tpr"][b]
        c[f"tprDC{suffix}"], c[f"tprBG{suffix}"] = r["defectcon"], r["bgad"]
        c[f"tprD{suffix}"], c[f"tprW{suffix}"], c[f"tprP{suffix}"] = r["mean"] * 100, wins(r["wins"]), r["t_p"]
    c["cross"] = region["roc_crossing"] * 100
    nat = analysis.natural_rate(region)
    c["natPDCar"], c["natPBGar"] = nat["precision"][0.01]["defectcon"], nat["precision"][0.01]["bgad"]
    c["natPer"] = nat["precision"][0.05]["defectcon"]
    c["dcAPnatr"], c["bgAPnatr"] = nat["ap_defectcon"], nat["ap_bgad"]

    # Tables 7-8: ablation
    ab = analysis.ablation()
    for code, cell in (("Az", "A0"), ("Do", "D1"), ("Po", "P1"), ("Jo", "J1"), ("Lo", "L1")):
        for name, k in (("AUROC", "auroc"), ("AP", "ap"), ("P", "pauc5")):
            c[f"ab{code}{name}"], c[f"ab{code}{name}sd"] = ab[cell][k]
        if cell == "A0":
            continue
        for name, k in (("AP", "ap"), ("P", "pauc5"), ("AUROC", "auroc")):
            r = ab[cell]["paired"][k]
            c[f"ab{code}D{name}"] = r["mean"]
            c[f"ab{code}D{name}r"] = r["mean"]
            c[f"ab{code}D{name}ci"] = (r["ci_lo"], r["ci_hi"])
            c[f"ab{code}D{name}w"] = wins(r["losses"])
            c[f"ab{code}D{name}p"] = r["wilcoxon_p"]
    c["abLoDAUROCr"] = abs(ab["L1"]["paired"]["auroc"]["mean"])

    # Appendix C: contrastive weight 0 (tuning-split L1)
    c["senLzeroP"] = next(r["pauc5"] for r in analysis.sensitivity() if r["point"] == "lcon0")

    # Section 5.5: virtual flaws
    base = analysis.summary("defectcon", "main", (42,))
    c["vfBaseAP"] = base["ap"][0]
    vf = analysis.summary("defectcon", "vf")
    for name, k in key.items():
        c[f"vfDC{name}"], c[f"vfDC{name}sd"] = vf[k]
    vft = analysis.summary("defectcon", "vf_tuned")
    for name in ("AP", "Pfive", "Pten", "AUROC"):
        c[f"vftDC{name}"], c[f"vftDC{name}sd"] = vft[key[name]]
    with open(results.METRICS_ROOT / "vf_tuned_operating_point.csv") as f:
        for r in csv.DictReader(f):
            name = {"recall": "Rec", "precision": "Prec", "f1": "Fone"}[r["metric"]]
            c[f"vftDC{name}"], c[f"vftDC{name}sd"] = float(r["mean"]), float(r["sd"])
    c["vfAUROCgainr"] = vf["auroc"][0] - base["auroc"][0]
    bg_base = analysis.summary("bgad", "main", (42,))["ap"][0]
    bg_vf = analysis.summary("bgad", "vf")["ap"][0]
    c["vfBGbase"], c["vfBGAP"] = bg_base, bg_vf
    c["vfBGtuneGain"] = analysis.summary("bgad", "vf_tuned")["ap"][0] - bg_vf
    c["vfDCshare"] = (base["ap"][0] - vf["ap"][0]) / base["ap"][0] * 100
    c["vfBGshare"] = (bg_base - bg_vf) / bg_base * 100
    c["vfdAPr"] = analysis.paired_vs(("defectcon", "vf"), ("bgad", "vf"), seeds=(42,))["ap"]["mean"]
    c["vfwAP"] = analysis.summary("defectcon", "vf_widened")["ap"][0]
    c["vfSCAUROCgain"] = (analysis.summary("supcon", "vf")["auroc"][0]
                          - analysis.summary("supcon", "main")["auroc"][0])
    return c


def _number(text: str) -> tuple[float, float]:
    """Parse a printed number; return (value, tolerance of one unit in the last digit)."""
    t = re.sub(r"\\[,;:!\s]", "", text).replace("$", "").strip()
    m = re.fullmatch(r"([+-]?)(\d+)(?:\.(\d+))?(?:\\times10\^\{?(-?\d+)\}?)?", t)
    if not m:
        raise ValueError(f"cannot parse {text!r}")
    sign, whole, frac, exp = m.groups()
    frac = frac or ""
    scale = 10.0 ** int(exp) if exp else 1.0
    value = float(f"{sign}{whole}.{frac or '0'}") * scale
    return value, 10.0 ** -len(frac) * scale


MACROS = paper_macros()


def test_every_body_macro_is_covered():
    missing = [m for m in MACROS if m not in computed() and m not in NOT_RECOMPUTED]
    assert not missing, f"no computation for: {missing}"


@pytest.mark.parametrize("name", sorted(m for m in MACROS if m not in NOT_RECOMPUTED))
def test_macro(name):
    got, want = computed()[name], MACROS[name]
    if isinstance(got, str):
        assert got == want
    elif isinstance(got, tuple):
        parts = re.sub(r"\\[,;:!\s]", "", want).strip("[] ").split(",")
        for g, w in zip(got, parts):
            value, tol = _number(w)
            assert abs(g - value) <= tol + 1e-12, f"{name}: {got} vs {want}"
    else:
        value, tol = _number(want)
        assert abs(got - value) <= tol + 1e-12, f"{name}: computed {got:.6g}, paper {want}"


# --------------------------------------------------------------------------
# Baseline operating-point cells (hard-coded in the tables, not macros)
# --------------------------------------------------------------------------
# Recall, precision and F1 at FPR <= 5% of every method use the threshold
# calibrated on each run's own validation normals (95th percentile), applied
# unchanged to the held-out set; mean and sample SD (n - 1) over the 10 runs.
BASELINE_CELLS = [  # (table label, column index after the metric name, method, condition)
    ("tab:main_comparison", 0, "supcon", "main"),
    ("tab:main_comparison", 1, "uninet", "main"),
    ("tab:vf_controlled", 0, "supcon", "vf"),
    ("tab:vf_controlled", 1, "uninet", "vf"),
    ("tab:vf_controlled", 2, "bgad", "vf"),
]
ROW_PREFIX = {"recall": "Recall at FPR", "precision": "Precision at FPR", "f1": "$F_1$ at FPR"}


def table_cell(label: str, row_prefix: str, column: int) -> tuple[str, str]:
    """(mean, sd) strings of a ``$m \\pm s$`` cell in the table with ``\\label{label}``."""
    text = paths.PAPER_NUMBERS.read_text(encoding="utf-8")
    start = text.index(f"\\label{{{label}}}")
    block = text[start:text.index("\\end{tabular}", start)]
    row = next(ln for ln in block.splitlines() if ln.strip().startswith(row_prefix))
    cell = row.split("&")[column + 1]
    m = re.search(r"([0-9.]+)\s*\\pm\s*([0-9.]+)", cell)
    return m.group(1), m.group(2)


@pytest.mark.parametrize("label,column,method,condition", BASELINE_CELLS)
@pytest.mark.parametrize("metric", ["recall", "precision", "f1"])
def test_baseline_operating_point(label, column, method, condition, metric):
    values = [m[metric] for m in results.collect(method, condition, (42,)).values()]
    mean, sd = stats.mean_sd(values)
    want_mean, want_sd = table_cell(label, ROW_PREFIX[metric], column)
    for got, want in ((mean, want_mean), (sd, want_sd)):
        value, tol = _number(want)
        assert abs(got - value) <= tol + 1e-12, \
            f"{method} {condition} {metric}: computed {mean:.4f} ± {sd:.4f}, paper {want_mean} ± {want_sd}"
