#!/usr/bin/env python3
"""Train DefectCon and score the trained model (Sections 4.6 and 5).

One run = one training seed on one split. Every run trains for the fixed step
budget, keeps the final model, and scores it on the split's validation lists
and on the held-out test set.

Main comparison (Table 3), 2 partition seeds x 5 folds x 3 training seeds:
    python scripts/train.py --split cv --partition-seed 100 101 --fold 0 1 2 3 4 \\
        --seeds 42 202 303 --hparams baseline
VF under baseline / VF-tuned / widened hyperparameters (Tables 9-10, Section 5.5):
    python scripts/train.py --split cv --condition vf --hparams baseline   ...
    python scripts/train.py --split cv --condition vf --hparams vf_tuned   ...
Ablation cell on the CV partitions (Tables 7-8):
    python scripts/train.py --split cv --partition-seed 100 101 --fold 0 1 2 3 4 --ablation L1
Tuning-split runs for the sensitivity sweep (Table C):
    python scripts/train.py --split tune --ablation A0
    python scripts/train.py --split tune --sensitivity temperature=0.05

Outputs per run under --out (default $DEFECTCON_OUT/<run group>/<run>/):
    model.pth, curve.csv, meta.json, scores_val.npz, scores_heldout.npz
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

from defectcon import engine, paths, scoring
from defectcon.protocol import RunConfig, load_hparams

SMOKE_TEST_STEPS = 6
SMOKE_TEST_BATCH = (4, 2)   # batch size, defects per batch


def run_group_name(a: argparse.Namespace) -> str:
    """Directory name shared by all runs of one configuration."""
    name = f"{a.split}_{a.condition}_{a.hparams}"
    if a.ablation:
        name += f"_{a.ablation}"
    if a.sensitivity:
        name += f"_{a.sensitivity[0]}={a.sensitivity[1]:g}"
    return name


def parse_point(text: str) -> tuple:
    axis, _, value = text.partition("=")
    if not value:
        raise argparse.ArgumentTypeError("expected AXIS=VALUE, e.g. temperature=0.05")
    return axis, float(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["cv", "tune"], required=True)
    ap.add_argument("--condition", choices=["novf", "vf"], default="novf")
    ap.add_argument("--partition-seed", type=int, nargs="+", default=[100, 101])
    ap.add_argument("--fold", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="training seeds (paper: 42 202 303 for the main comparison)")
    ap.add_argument("--hparams", default="baseline",
                    help="name in configs/hparams/ (baseline, vf_tuned, vf_widened) or a YAML path")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--ablation", choices=["A0", "D1", "P1", "J1", "L1"])
    group.add_argument("--sensitivity", type=parse_point, metavar="AXIS=VALUE",
                       help="one point of configs/sensitivity.yaml, e.g. batch_size=64")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true", help="print the run schedules and exit")
    ap.add_argument("--smoke-test", action="store_true",
                    help=f"pipeline check: {SMOKE_TEST_STEPS} steps with batch size "
                         f"{SMOKE_TEST_BATCH[0]} ({SMOKE_TEST_BATCH[1]} defects)")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="data-loader workers (default: configs/defectcon.yaml)")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    hparams = load_hparams(a.hparams)
    out_root = a.out or paths.OUT_ROOT / run_group_name(a)
    device = torch.device(a.device)
    if a.split == "tune":
        jobs = [(None, None, s) for s in a.seeds]
    else:
        jobs = [(ps, k, s) for ps in a.partition_seed for k in a.fold for s in a.seeds]

    for pseed, fold, seed in jobs:
        cfg = RunConfig.build(hparams, condition=a.condition, seed=seed, ablation=a.ablation,
                              sensitivity=a.sensitivity,
                              budget_steps=SMOKE_TEST_STEPS if a.smoke_test else None)
        if a.smoke_test:
            cfg.batch_size, cfg.hparams["defects_per_batch"] = SMOKE_TEST_BATCH
        split_dir = paths.split_dir(a.split, a.condition, pseed, fold)
        run = f"p{pseed}_f{fold}_s{seed}" if a.split == "cv" else f"s{seed}"
        run_dir = out_root / run
        if a.dry_run:
            n = len(paths.read_lines(split_dir / "train_normal.txt"))
            print(run_dir, cfg.schedule(n).__dict__)
            continue
        if (run_dir / "meta.json").exists():
            print(f"{run_dir}: already complete, skipped")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(run_dir / "train.log")
        logging.getLogger().addHandler(handler)
        t0 = time.time()
        curve: list = []

        def on_epoch(epoch: int, record: dict) -> None:
            curve.append(record)
            engine.write_curve(run_dir / "curve.csv", curve)

        model, history, info = engine.train(cfg, split_dir, device, on_epoch=on_epoch,
                                            num_workers=a.num_workers)
        hours = (time.time() - t0) / 3600
        engine.save_checkpoint(run_dir / "model.pth", model, cfg, info["schedule"], len(history))
        for name, s in scoring.score_split(model, split_dir, device).items():
            scoring.save_scores(run_dir / f"scores_{name}.npz", s)
        meta = {"split": a.split, "condition": a.condition, "partition_seed": pseed, "fold": fold,
                "seed": seed, "hparams_name": a.hparams, "hparams": cfg.hparams, "architecture": cfg.architecture,
                "ablation": a.ablation, "sensitivity": a.sensitivity, "smoke_test": a.smoke_test,
                "epochs_trained": len(history), "total_hours": round(hours, 2),
                "final_val_ap": history[-1]["val_ap"], "final_val_auroc": history[-1]["val_auroc"],
                **info}
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        logging.getLogger().removeHandler(handler)
        print(f"{run_dir}: done in {hours:.2f} h")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
