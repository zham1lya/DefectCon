#!/usr/bin/env python3
"""Fig. 7 and the silhouette scores of Appendix D.

Embeddings are the L2-normalised pooled 512-d encoder features of the 5,150
held-out images. The silhouette uses the cosine metric in the 512-d space with
the class labels as clusters. UMAP: n_neighbors=30, min_dist=0.1, cosine
metric, 2 components, random_state=42 (display only).

    # silhouette of several models (results/metrics/silhouette.csv holds the ten seed-42 models)
    python scripts/embeddings.py silhouette --checkpoints runs/cv_novf_baseline/p*_s42/model.pth
    # Fig. 7 for the representative model (partition 101 / fold 0)
    python scripts/embeddings.py umap --checkpoint runs/cv_novf_baseline/p101_f0_s42/model.pth
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import silhouette_score

from defectcon import paths, scoring


def heldout_embeddings(checkpoint: Path, device: torch.device):
    model, _ = scoring.load_checkpoint(checkpoint, device)
    held = paths.heldout_images()
    z = scoring.embed_images(model, held["normal"] + held["defect"], device)
    y = np.array([0] * len(held["normal"]) + [1] * len(held["defect"]))
    return z, y


def plot_umap(z: np.ndarray, y: np.ndarray, sil: float, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    emb = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", n_components=2,
                    random_state=42).fit_transform(z)
    fig, ax = plt.subplots(figsize=(7.1, 6.3))
    n, d = y == 0, y == 1
    ax.scatter(emb[n, 0], emb[n, 1], s=4, c="#1f77b4", alpha=0.25, linewidths=0, label="normal", zorder=2)
    ax.scatter(emb[d, 0], emb[d, 1], s=38, c="#d95f02", alpha=0.95, linewidths=0, label="defect", zorder=3)
    ax.set_title("UMAP of held-out embeddings, DefectCon (no VF), representative partition", fontsize=11)
    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.grid(True, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=12)
    leg.legend_handles[0].set_sizes([22])
    ax.text(0.015, 0.975, f"Silhouette (512-d embedding, cosine) = {sil:.3f}", transform=ax.transAxes,
            va="top", ha="left", family="monospace", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#aaaaaa"), zorder=5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("silhouette")
    s.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    s.add_argument("--out", type=Path, default=paths.OUT_ROOT / "silhouette.csv")
    u = sub.add_parser("umap")
    u.add_argument("--checkpoint", type=Path, required=True)
    u.add_argument("--out", type=Path, default=paths.FIGURES_ROOT / "fig7_umap.png")
    for p in (s, u):
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    if a.cmd == "umap":
        z, y = heldout_embeddings(a.checkpoint, device)
        sil = float(silhouette_score(z, y, metric="cosine"))
        print(f"silhouette = {sil:.4f}")
        plot_umap(z, y, sil, a.out)
        return
    rows = []
    for c in a.checkpoints:
        z, y = heldout_embeddings(c, device)
        rows.append({"checkpoint": str(c), "silhouette_cosine": float(silhouette_score(z, y, metric="cosine"))})
        print(rows[-1])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
