#!/usr/bin/env python3
"""Fig. 5: Grad-CAM maps for representative held-out images (Section 5.7, Appendix D).

Model: the seed-42 model of partition 101 / fold 0 (held-out AP closest to the
seed-42 mean). Maps are computed at the encoder output (512 channels, 40x40),
for the defect class: channel weights are the spatial mean of the gradients,
the weighted sum is rectified, min-max normalised per image and bilinearly
upsampled to the input size. The images are listed by ID in
``results/figure_inputs/gradcam_ids.csv``; the IDs are resolved to image files
with the private mapping ``private/id_map.csv``.

    python scripts/gradcam.py --checkpoint <run dir>/model.pth
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from defectcon import paths, scoring
from defectcon.data import load_grayscale
from defectcon.transforms import eval_transform


def gradcam(model, x: torch.Tensor, target_class: int = 1):
    store = {}

    def hook(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = model.encoder.register_forward_hook(hook)
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    probs = F.softmax(logits.float(), dim=1)[:, target_class].detach().cpu().numpy()
    model.zero_grad(set_to_none=True)
    logits[:, target_class].sum().backward()
    handle.remove()
    act, grad = store["act"], store["grad"]
    cam = F.relu((grad.mean(dim=(2, 3), keepdim=True) * act).sum(1))
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return cam.detach().cpu().numpy(), probs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--images", type=Path,
                    default=paths.RESULTS_ROOT / "figure_inputs" / "gradcam_ids.csv")
    ap.add_argument("--id-map", type=Path, default=paths.REPO_ROOT / "private" / "id_map.csv")
    ap.add_argument("--out", type=Path, default=paths.FIGURES_ROOT / "fig5_gradcam.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    device = torch.device(a.device)
    model, _ = scoring.load_checkpoint(a.checkpoint, device)
    rows = list(csv.DictReader(open(a.images)))
    names = {r["image_id"]: r["original_name"] for r in csv.DictReader(open(a.id_map))}
    image_paths = [paths.resolve_heldout([names[r["image_id"]]], r["class"])[0] for r in rows]
    transform = eval_transform()
    x = torch.stack([transform(load_grayscale(p)) for p in image_paths]).to(device)
    cams, probs = gradcam(model, x)
    gray = (x.detach().cpu() * 0.5 + 0.5).clamp(0, 1)[:, 0].numpy()
    for r, p in zip(rows, probs):
        print(f"{r['row']:27s} P(defect) = {p:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(13.1, 10.2))
    for i in range(len(rows)):
        r, c = divmod(i, 2)
        ax_in, ax_cam = axes[r, 2 * c], axes[r, 2 * c + 1]
        ax_in.imshow(gray[i], cmap="gray")
        ax_in.set_title(f"input\nP(def)={probs[i]:.2f}", fontsize=10)
        ax_cam.imshow(gray[i], cmap="gray")
        ax_cam.imshow(cams[i], cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax_cam.set_title("Grad-CAM", fontsize=10)
        for ax in (ax_in, ax_cam):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        if c == 0:
            ax_in.set_ylabel(rows[i]["row"], fontsize=13)
    fig.suptitle("DefectCon: Grad-CAM on held-out radiographs (representative partition; "
                 "qualitative, not pixel localisation)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
