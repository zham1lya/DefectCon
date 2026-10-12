#!/usr/bin/env python3
"""Score saved checkpoints on a validation split and on the held-out test set.

Inference only. Arrays ``path`` (basename), ``label`` and ``score`` (softmax
defect probability, bilinear preprocessing) are written per checkpoint.

Training runs of ``scripts/train.py`` (``--runs``): the split, condition and
hyperparameters are read from each run's ``meta.json`` and the scores are
written into the results layout read by ``aggregate.py`` and ``analyze.py``,
``<results>/scores/defectcon/<condition>/p<P>_f<F>_s<S>_{val,heldout}.npz``
(condition ``main``, ``vf``, ``vf_tuned`` or ``vf_widened``). Existing files
are not overwritten without ``--overwrite``:

    python scripts/score.py --runs $DEFECTCON_OUT/cv_novf_baseline/p*_f*_s* \
        --results-root my_results

Any checkpoint (``--checkpoints``): writes ``<out>/<name>_val.npz`` and
``<out>/<name>_heldout.npz``.

``--split`` selects the validation lists: ``tune`` for the tuning split, or
``cv:<partition_seed>:<fold>`` for a CV fold (the VF and non-VF copies of a
fold share the same validation lists).

    python scripts/score.py --checkpoints runs/cv_novf_baseline/p100_f0_s42/model.pth \\
        --split cv:100:0 --out results/scores/defectcon/main --name p100_f0_s42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from defectcon import paths, results, scoring


def parse_split(text: str) -> Path:
    if text == "tune":
        return paths.split_dir("tune")
    kind, _, rest = text.partition(":")
    if kind == "cv" and rest.count(":") == 1:
        seed, fold = rest.split(":")
        return paths.split_dir("cv", "novf", int(seed), int(fold))
    raise argparse.ArgumentTypeError("expected 'tune' or 'cv:<partition_seed>:<fold>'")


# (condition, hyperparameters) of a training run -> condition in results/scores
RESULT_CONDITION = {("novf", "baseline"): "main", ("vf", "baseline"): "vf",
                    ("vf", "vf_tuned"): "vf_tuned", ("vf", "vf_widened"): "vf_widened"}


def run_target(run_dir: Path, results_root: Path) -> tuple:
    """(split directory, output stem) of a training run in the results layout."""
    meta = json.loads((run_dir / "meta.json").read_text())
    # runs written before hparams_name was recorded: <split>_<condition>_<hparams>
    key = (meta["condition"], meta.get("hparams_name") or run_dir.parent.name.split("_", 2)[-1])
    if meta["partition_seed"] is None or meta["ablation"] or meta["sensitivity"] or key not in RESULT_CONDITION:
        raise SystemExit(f"{run_dir}: only CV runs of the main and VF comparisons have a place in "
                         f"results/scores; score other runs with --checkpoints")
    ps, k, s = meta["partition_seed"], meta["fold"], meta["seed"]
    stem = results.score_file("defectcon", RESULT_CONDITION[key], (ps, k, s), "", results_root / "scores")
    return paths.split_dir("cv", "novf", ps, k), str(stem)[:-len("_.npz")]


def score(ckpt: Path, split: Path, stem: str, device: torch.device) -> None:
    model, _ = scoring.load_checkpoint(ckpt, device)
    for part, s in scoring.score_split(model, split, device).items():
        scoring.save_scores(Path(f"{stem}_{part}.npz"), s)
    print(f"{ckpt} -> {stem}_{{val,heldout}}.npz")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs", type=Path, nargs="+", help="run directories written by train.py")
    src.add_argument("--checkpoints", type=Path, nargs="+")
    ap.add_argument("--results-root", type=Path, default=paths.RESULTS_ROOT,
                    help="with --runs: results tree to write into (default: $DEFECTCON_RESULTS or results/)")
    ap.add_argument("--overwrite", action="store_true", help="with --runs: replace existing score files")
    ap.add_argument("--split", type=parse_split, help="with --checkpoints")
    ap.add_argument("--out", type=Path, help="with --checkpoints")
    ap.add_argument("--name", nargs="+", default=None,
                    help="with --checkpoints: output name per checkpoint (default: the parent directory name)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    if a.runs:
        jobs = [(r / "model.pth", *run_target(r, a.results_root)) for r in a.runs]
        existing = [stem for _, _, stem in jobs if Path(f"{stem}_heldout.npz").exists()]
        if existing and not a.overwrite:
            ap.error(f"{len(existing)} score files exist (first: {existing[0]}_heldout.npz); "
                     f"use --overwrite or another --results-root")
    else:
        if a.split is None or a.out is None:
            ap.error("--checkpoints needs --split and --out")
        names = a.name or [c.parent.name for c in a.checkpoints]
        if len(names) != len(a.checkpoints):
            ap.error("--name needs one entry per checkpoint")
        jobs = [(c, a.split, str(a.out / n)) for c, n in zip(a.checkpoints, names)]
    for ckpt, split, stem in jobs:
        score(ckpt, split, stem, device)


if __name__ == "__main__":
    main()
