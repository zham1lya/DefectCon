#!/usr/bin/env python3
"""Figures 3 and 4 from the released results (no GPU, no images).

    python scripts/figures.py operating-region   # Fig. 3: mean ROC and paired pAUC difference
    python scripts/figures.py convergence        # Fig. 4: training loss, validation AP and AUROC

Writes PNG (and PDF for Fig. 3) to ``figures/``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from defectcon import analysis, paths, results  # noqa: E402
from defectcon.protocol import load_hparams  # noqa: E402

DC, BG = "#C8501E", "#2B6CB0"
OPERATING_FPR = 0.05


def operating_region(out: Path) -> None:
    r = analysis.operating_region()
    grid = r["roc"]["grid"]
    dc_m, dc_s = r["roc"]["defectcon_mean"], r["roc"]["defectcon_sd"]
    bg_m, bg_s = r["roc"]["bgad_mean"], r["roc"]["bgad_sd"]
    bounds = np.array(sorted(r["pauc"]))
    diff = np.array([r["pauc"][b]["mean"] for b in bounds])
    dsd = np.array([r["pauc"][b]["sd"] for b in bounds])
    wins = np.array([r["pauc"][b]["wins"] for b in bounds])

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.0))
    a = ax[0]
    a.plot(grid * 100, dc_m, color=DC, lw=2.1, label="DefectCon", zorder=3)
    a.fill_between(grid * 100, dc_m - dc_s, dc_m + dc_s, color=DC, alpha=0.16, lw=0)
    a.plot(grid * 100, bg_m, color=BG, lw=2.1, label="BGAD", zorder=3)
    a.fill_between(grid * 100, bg_m - bg_s, bg_m + bg_s, color=BG, alpha=0.16, lw=0)
    cross = r["roc_crossing"] * 100
    a.axvline(OPERATING_FPR * 100, ls="--", color="0.35", lw=1.2, zorder=1)
    a.annotate(f"crossing\n{cross:.2f}%", xy=(cross, np.interp(cross / 100, grid, dc_m)),
               xytext=(cross + 2.4, 0.47), fontsize=9, color="0.25",
               arrowprops=dict(arrowstyle="->", color="0.45", lw=1.0))
    a.text(OPERATING_FPR * 100 + 0.12, 0.055, "FPR = 5%", fontsize=9, color="0.35", rotation=90)
    a.set_xlim(0, 15); a.set_ylim(0, 0.9)
    a.set_xlabel("False positive rate (%)")
    a.set_ylabel("True positive rate")
    a.set_title("(a)  Mean ROC, held-out test set", fontsize=11)
    a.grid(alpha=0.25, lw=0.6)
    a.legend(loc="lower right", frameon=True, framealpha=0.95)

    b_ = ax[1]
    b_.axhline(0.0, color="0.35", lw=1.2, zorder=1)
    b_.fill_between(bounds * 100, diff - dsd, diff + dsd, color=DC, alpha=0.18, lw=0,
                    label="$\\pm$1 SD over 10 partitions")
    b_.plot(bounds * 100, diff, color=DC, lw=2.1, zorder=3,
            label="pAUC$_{\\mathrm{DefectCon}}-$pAUC$_{\\mathrm{BGAD}}$")
    sig = wins == 10
    b_.scatter(bounds[sig] * 100, diff[sig], s=26, color=DC, zorder=4,
               edgecolor="white", linewidth=0.7, label="10/10 partitions (p = 0.0020)")
    b_.axvline(OPERATING_FPR * 100, ls="--", color="0.35", lw=1.2, zorder=1)
    b_.text(OPERATING_FPR * 100 + 0.12, -0.0075, "FPR = 5%", fontsize=9, color="0.35", rotation=90)
    i = int(np.argmax(diff < 0))
    zero = (bounds[i - 1] + (bounds[i] - bounds[i - 1]) * (diff[i - 1] / (diff[i - 1] - diff[i]))) * 100
    b_.annotate(f"sign change\n{zero:.1f}%", xy=(zero, 0), xytext=(zero - 0.4, 0.0115),
                fontsize=9, color="0.25", ha="right",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=1.0))
    b_.set_xlim(1, 15); b_.set_ylim(-0.011, 0.047)
    b_.set_xlabel("Upper FPR bound of the partial AUC (%)")
    b_.set_ylabel("Paired difference in McClish-standardized pAUC")
    b_.set_title("(b)  Paired pAUC difference vs upper bound", fontsize=11)
    b_.grid(alpha=0.25, lw=0.6)
    b_.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.95)
    fig.suptitle("DefectCon vs BGAD at low false-alarm budgets (3 training seeds, 10 partitions)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(out / f"fig3_operating_region.{ext}", dpi=200, bbox_inches="tight", metadata=meta)
    print(f"wrote {out}/fig3_operating_region.png/.pdf (ROC crossing {cross:.2f}%)")


def convergence(out: Path) -> None:
    """Per-fold curves (thin), partition-seed means (solid) and overall mean (dashed)."""
    w_con = load_hparams("baseline")["contrastive_weight"]
    curves = {}
    for ps, k in results.partitions():
        rows = list(csv.DictReader(open(paths.RESULTS_ROOT / "curves" / "defectcon" / "main" / f"p{ps}_f{k}_s42.csv")))
        curves[(ps, k)] = {
            "loss": np.array([w_con * float(r["train_loss_con"]) + float(r["train_loss_ce"]) for r in rows]),
            "val_ap": np.array([float(r["val_ap"]) for r in rows]),
            "val_auroc": np.array([float(r["val_auroc"]) for r in rows])}
    n = min(len(c["loss"]) for c in curves.values())
    epochs = np.arange(1, n + 1)
    colors = {100: "#0072B2", 101: "#D55E00"}
    thin = {100: "#56A0D3", 101: "#F5A06A"}
    panels = (("loss", "A  Training convergence", "Training loss  (w$\\cdot$SupCon + focal)", None),
              ("val_auroc", "B  Validation AUROC", "Validation AUROC", (0.5, 1.0)),
              ("val_ap", "C  Validation AP", "Validation AP", (0.0, 0.8)))
    fig, ax = plt.subplots(1, 3, figsize=(15.4, 4.6))
    for a, (key, title, ylabel, ylim) in zip(ax, panels):
        for i, ps in enumerate(results.PARTITION_SEEDS):
            mats = np.array([curves[(ps, k)][key][:n] for k in results.FOLDS])
            for m in mats:
                a.plot(epochs, m, color=thin[ps], lw=1.0, alpha=0.8)
            a.plot(epochs, mats.mean(0), color=colors[ps], lw=2.6,
                   label=f"Repeat {i + 1} (mean of 5 folds)")
        a.plot(epochs, np.array([c[key][:n] for c in curves.values()]).mean(0),
               color="0.1", lw=2.2, ls="--", label="Overall mean (10 folds)")
        a.set_title(title, loc="left", fontsize=12, fontweight="bold")
        a.set_xlabel("Epoch"); a.set_ylabel(ylabel)
        if ylim:
            a.set_ylim(*ylim)
        a.grid(alpha=0.6, lw=0.6, color="0.85")
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    ax[0].legend(frameon=False)
    fig.suptitle("DefectCon (no VF): convergence across repeated 5-fold CV (2 repeats x 5 folds)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig4_convergence.png", dpi=200)
    print(f"wrote {out}/fig4_convergence.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure", choices=["operating-region", "convergence"])
    ap.add_argument("--out", type=Path, default=paths.FIGURES_ROOT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    {"operating-region": operating_region, "convergence": convergence}[a.figure](a.out)


if __name__ == "__main__":
    main()
