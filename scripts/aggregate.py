#!/usr/bin/env python3
"""Absolute results from the released scores and metric files.

    python scripts/aggregate.py --condition main      # Table 3 (30 runs per method)
    python scripts/aggregate.py --condition vf        # Table 9
    python scripts/aggregate.py --condition vf_tuned  # Table 10
    python scripts/aggregate.py --condition vf_widened
    python scripts/aggregate.py --condition ablation  # Tables 7-8
    python scripts/aggregate.py --condition sensitivity  # Table C
    python scripts/aggregate.py --condition per_partition  # Appendix E

Writes ``results/metrics/<condition>_summary.csv`` (or the table itself).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from defectcon import analysis, results

SEEDS = {"main": (42, 202, 303), "vf": (42,), "vf_tuned": (42,), "vf_widened": (42,)}
METHODS = {"main": ("defectcon", "bgad"), "vf": ("defectcon", "bgad"),
           "vf_tuned": ("defectcon", "bgad"), "vf_widened": ("defectcon",)}


def write(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True,
                    choices=list(SEEDS) + ["ablation", "sensitivity", "per_partition"])
    ap.add_argument("--out", type=Path, default=results.METRICS_ROOT)
    a = ap.parse_args()
    c = a.condition
    if c in SEEDS:
        rows = []
        for m in METHODS[c]:
            for metric, (mean, sd) in analysis.summary(m, c, SEEDS[c]).items():
                rows.append({"method": m, "metric": metric, "mean": round(mean, 6), "sd": round(sd, 6)})
                print(f"{m:10s} {metric:9s} {mean:.4f} ± {sd:.4f}")
        write(a.out / f"{c}_summary.csv", rows)
    elif c == "ablation":
        rows = []
        for cell, v in analysis.ablation().items():
            for metric in ("auroc", "ap", "pauc5"):
                row = {"cell": cell, "metric": metric, "mean": round(v[metric][0], 6), "sd": round(v[metric][1], 6)}
                if "paired" in v:
                    p = v["paired"][metric]
                    row.update(delta=round(p["mean"], 6), delta_sd=round(p["sd"], 6), ci_lo=round(p["ci_lo"], 6),
                               ci_hi=round(p["ci_hi"], 6), a0_better=p["losses"], wilcoxon_p=round(p["wilcoxon_p"], 6))
                rows.append(row)
        rows = [{k: r.get(k, "") for k in ["cell", "metric", "mean", "sd", "delta", "delta_sd", "ci_lo",
                                          "ci_hi", "a0_better", "wilcoxon_p"]} for r in rows]
        write(a.out / "ablation_summary.csv", rows)
    elif c == "sensitivity":
        write(a.out / "sensitivity_summary.csv", analysis.sensitivity())
    else:
        write(a.out / "main_per_partition_ap.csv",
              [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
               for r in analysis.per_partition_ap()])


if __name__ == "__main__":
    main()
