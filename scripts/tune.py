#!/usr/bin/env python3
"""Hyperparameter search on the tuning split (Appendix B).

Protocol: 20 trials per study; the first 7 are a seeded Latin-hypercube design
enqueued before any worker starts; later trials use TPE
(``TPESampler(seed=42 + worker_index)``); a median pruner stops unpromising
trials after a warm-up. Trials report the trailing-5 smoothed validation AP
every 495-step block; the objective is its maximum within the fixed budget.
Two workers share one Optuna journal.

    python scripts/tune.py --search-space defectcon_baseline --enqueue-only
    CUDA_VISIBLE_DEVICES=0 python scripts/tune.py --search-space defectcon_baseline --worker-index 0
    CUDA_VISIBLE_DEVICES=1 python scripts/tune.py --search-space defectcon_baseline --worker-index 1
    python scripts/tune.py --search-space defectcon_baseline --export-trials

Search spaces: configs/search/{defectcon_baseline,defectcon_vf,defectcon_vf_widened}.yaml.
The journals of the reported searches are in results/tuning/.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from scipy.stats import qmc

from defectcon import engine, paths
from defectcon.protocol import RunConfig, load_yaml, protocol

SEED = 42


def load_space(name: str) -> Dict[str, Any]:
    return load_yaml(paths.CONFIGS_ROOT / "search" / f"{name}.yaml")


def lhs_starts(space: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Seeded Latin-hypercube starting points; the last column selects defects per batch."""
    params, ratios = space["space"], space["defects_per_batch"]
    u = qmc.LatinHypercube(d=len(params) + 1, seed=SEED).random(space["n_startup_trials"])
    starts = []
    for row in u:
        cfg = {}
        for (name, (lo, hi, log)), x in zip(params.items(), row[:-1]):
            cfg[name] = (float(10 ** (math.log10(lo) + x * (math.log10(hi) - math.log10(lo))))
                         if log else float(lo + x * (hi - lo)))
        cfg["defects_per_batch"] = ratios[min(len(ratios) - 1, int(row[-1] * len(ratios)))]
        starts.append(cfg)
    return starts


def get_study(space: Dict[str, Any], journal: Path, worker_index: int = 0) -> optuna.Study:
    journal.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(journal)))
    return optuna.create_study(
        study_name=space["study_name"], storage=storage, direction="maximize",
        sampler=TPESampler(seed=SEED + worker_index, n_startup_trials=space["n_startup_trials"]),
        pruner=MedianPruner(n_startup_trials=space["n_startup_trials"],
                            n_warmup_steps=space["pruner_warmup_blocks"]),
        load_if_exists=True)


def objective(trial: optuna.Trial, space: Dict[str, Any], device: torch.device,
              budget_steps: int | None) -> float:
    hp = {name: trial.suggest_float(name, lo, hi, log=log) for name, (lo, hi, log) in space["space"].items()}
    hp["defects_per_batch"] = trial.suggest_categorical("defects_per_batch", space["defects_per_batch"])
    cfg = RunConfig.build(hp, condition=space["condition"], seed=SEED, budget_steps=budget_steps)
    split = paths.split_dir("tune", space["condition"])
    sched = cfg.schedule(len(paths.read_lines(split / "train_normal.txt")))
    trial.set_user_attr("epochs", sched.epochs)
    trial.set_user_attr("steps_per_epoch", sched.steps_per_epoch)
    trial.set_user_attr("total_steps", sched.total_steps)
    block = protocol()["step_block"]
    state = {"best": -1.0, "last": -1}

    def on_epoch(epoch: int, record: Dict[str, Any]) -> None:
        state["best"] = max(state["best"], record["val_ap_smooth5"])
        key = ((epoch + 1) * sched.steps_per_epoch) // block
        if key > state["last"]:
            state["last"] = key
            trial.report(record["val_ap_smooth5"], step=key)
            if trial.should_prune():
                raise optuna.TrialPruned()

    engine.train(cfg, split, device, on_epoch=on_epoch)
    return state["best"]


def export_trials(study: optuna.Study, out: Path) -> None:
    keys = sorted({k for t in study.trials for k in t.params})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["number", "state", "value", *keys])
        w.writeheader()
        for t in study.trials:
            w.writerow({"number": t.number, "state": t.state.name, "value": t.value, **t.params})
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search-space", required=True,
                    choices=sorted(p.stem for p in (paths.CONFIGS_ROOT / "search").glob("defectcon_*.yaml")))
    ap.add_argument("--journal", type=Path, default=None,
                    help="default: $DEFECTCON_OUT/tuning/<study_name>.journal")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--enqueue-only", action="store_true", help="enqueue the Latin-hypercube starts and exit")
    mode.add_argument("--export-trials", action="store_true", help="write the trials of the journal to CSV")
    ap.add_argument("--worker-index", type=int, default=0, help="must differ between parallel workers")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke-test", action="store_true", help="2 trials of 6 steps each")
    a = ap.parse_args()

    space = load_space(a.search_space)
    journal = a.journal or paths.OUT_ROOT / "tuning" / f"{space['study_name']}.journal"
    study = get_study(space, journal, a.worker_index)
    if a.export_trials:
        export_trials(study, journal.with_name(f"{space['study_name']}_trials.csv"))
        return
    if a.enqueue_only:
        if study.trials:
            print("study already has trials; nothing enqueued")
            return
        for cfg in lhs_starts(space):
            study.enqueue_trial(cfg)
        print(f"enqueued {space['n_startup_trials']} starting points into {journal}")
        return
    if not study.trials:
        raise SystemExit("no trials enqueued; run with --enqueue-only first")
    n_trials = 2 if a.smoke_test else space["n_trials"]
    budget = 6 if a.smoke_test else None
    device = torch.device(a.device)
    while sum(t.state != optuna.trial.TrialState.WAITING for t in study.get_trials()) < n_trials:
        study.optimize(lambda t: objective(t, space, device, budget), n_trials=1)


if __name__ == "__main__":
    main()
