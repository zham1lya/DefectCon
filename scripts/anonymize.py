#!/usr/bin/env python3
"""Replace original image names with neutral IDs in the released files.

Real images are ``img_000001...`` (sorted by original name) and synthetic VF
images ``vf_000001...``; the mapping is read from ``private/id_map.csv``, which is
not released. Row order is never changed, so every metric computed from the
released files is unchanged.

The script

* rewrites the ``path`` array of every score file in ``results/scores`` to IDs,
* writes ``results/figure_inputs/gradcam_ids.csv`` from the private image list,
* writes the released split definitions ``results/splits/assignments.csv`` from
  the private split lists (``--splits`` and, for the VF copies, ``--vf-splits``),
* replaces absolute paths in JSON files under ``results/`` by their last component.

    python scripts/anonymize.py --splits $DEFECTCON_SPLITS --vf-splits <dir with cv_seed*_vf>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np

from defectcon import paths

ID_MAP = paths.REPO_ROOT / "private" / "id_map.csv"
SPLIT_SETS = {"heldout": None, "tune": "tune", "p100": "cv_seed100", "p101": "cv_seed101"}


def load_map(path: Path) -> Dict[str, Dict[str, str]]:
    """``{"real": {name: id}, "synthetic": {name: id}}``; real names also without extension."""
    out: Dict[str, Dict[str, str]] = {"real": {}, "synthetic": {}}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["kind"]][r["original_name"]] = r["image_id"]
            if r["kind"] == "real":
                out["real"][os.path.splitext(r["original_name"])[0]] = r["image_id"]
    return out


def real_id(name: str, ids: Dict[str, Dict[str, str]]) -> str:
    name = os.path.basename(str(name))
    if name.startswith("img_"):
        return name
    return ids["real"][name]


def anonymize_scores(ids) -> int:
    n = 0
    for f in sorted((paths.RESULTS_ROOT / "scores").rglob("*.npz")):
        z = np.load(f, allow_pickle=True)
        new = np.array([real_id(p, ids) for p in z["path"]])
        if not np.array_equal(new, z["path"]):
            np.savez_compressed(f, path=new, label=z["label"], score=z["score"])
            n += 1
    return n


def gradcam_ids(ids, private_list: Path) -> None:
    out = paths.RESULTS_ROOT / "figure_inputs" / "gradcam_ids.csv"
    with open(private_list) as f:
        rows = [{"row": r["row"], "image_id": real_id(r["image"], ids), "class": r["class"]}
                for r in csv.DictReader(f)]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row", "image_id", "class"])
        w.writeheader()
        w.writerows(rows)


def split_assignments(ids, splits: Path, vf_splits: Path) -> int:
    """One row per image: class, held-out membership, tuning role, fold roles, VF use."""
    table: Dict[str, Dict[str, str]] = {}

    def row(image_id: str, cls: str) -> Dict[str, str]:
        return table.setdefault(image_id, {"image_id": image_id, "class": cls, "heldout": "0",
                                           "tune": "", "p100": "", "p101": "", "synthetic": "0"})

    for kind, fname in (("normal", "heldout_normal.txt"), ("defect", "heldout_defective.txt")):
        for name in paths.read_lines(splits / fname):
            row(real_id(name, ids), kind)["heldout"] = "1"
    for key, sub in (("tune", "tune"),):
        for part in ("train", "val"):
            for kind in ("normal", "defect"):
                for p in paths.read_lines(splits / sub / f"{part}_{kind}.txt"):
                    row(real_id(p, ids), kind)[key] = part
    for key, sub in (("p100", "cv_seed100"), ("p101", "cv_seed101")):
        for k in range(5):
            for kind in ("normal", "defect"):
                for p in paths.read_lines(splits / sub / f"fold{k}" / f"val_{kind}.txt"):
                    row(real_id(p, ids), kind)[key] = f"val{k}"
                for p in paths.read_lines(splits / sub / f"fold{k}" / f"train_{kind}.txt"):
                    r = row(real_id(p, ids), kind)
                    r[key] = r[key] or "train"
    synth = set()
    for p in paths.read_lines(vf_splits / "cv_seed100_vf" / "fold0" / "train_defect.txt"):
        if paths.is_synthetic(p):
            synth.add(ids["synthetic"][os.path.basename(p)])
    for s in sorted(synth):
        r = row(s, "defect")
        r.update(synthetic="1", tune="train", p100="train", p101="train")
    out = paths.RESULTS_ROOT / "splits" / "assignments.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "class", "heldout", "tune", "p100", "p101", "synthetic"])
        w.writeheader()
        w.writerows(sorted(table.values(), key=lambda r: r["image_id"]))
    return len(table)


def strip_json_paths() -> int:
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        if isinstance(v, str) and v.startswith("/"):
            return os.path.basename(v.rstrip("/"))
        return v

    n = 0
    for f in sorted(paths.RESULTS_ROOT.rglob("*.json")):
        d = json.loads(f.read_text())
        c = clean(d)
        if c != d:
            f.write_text(json.dumps(c, indent=1))
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, default=paths.SPLITS_ROOT)
    ap.add_argument("--vf-splits", type=Path, default=paths.SPLITS_ROOT)
    ap.add_argument("--gradcam-list", type=Path, default=paths.REPO_ROOT / "private" / "gradcam_images.csv")
    a = ap.parse_args()
    ids = load_map(ID_MAP)
    print(f"score files rewritten: {anonymize_scores(ids)}")
    gradcam_ids(ids, a.gradcam_list)
    print(f"split assignments: {split_assignments(ids, a.splits, a.vf_splits)} images")
    print(f"JSON files with absolute paths cleaned: {strip_json_paths()}")


if __name__ == "__main__":
    main()
