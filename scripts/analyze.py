#!/usr/bin/env python3
"""Paired comparisons and derived statistics (Section 5).

    python scripts/analyze.py paired            # Table 4, DefectCon vs BGAD (n = 10 partitions)
    python scripts/analyze.py paired-vf         # Section 5.5, DefectCon vs BGAD under VF
    python scripts/analyze.py variance          # Table 5
    python scripts/analyze.py per-seed          # Section 5.2.2
    python scripts/analyze.py ensembles         # Section 5.2.3 and Tables 9-10
    python scripts/analyze.py operating-region  # Table 6 and the data of Fig. 3
    python scripts/analyze.py natural-rate      # Section 5.3

Writes JSON files to ``results/metrics/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from defectcon import analysis, results


def to_json(obj):
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def compute(name: str):
    if name == "paired":
        return analysis.paired_vs(("defectcon", "main"), ("bgad", "main"))
    if name == "paired-vf":
        return {c: analysis.paired_vs(("defectcon", c), ("bgad", "vf"), seeds=(42,))
                for c in ("vf", "vf_widened")}
    if name == "variance":
        return {m: {k: analysis.variance(m, k) for k in analysis.METRIC_KEYS} for m in ("defectcon", "bgad")}
    if name == "per-seed":
        return analysis.per_seed_ap()
    if name == "ensembles":
        return {"main": {m: analysis.ensembles(m, "main") for m in ("defectcon", "bgad")},
                "vf": analysis.ensembles("defectcon", "vf", (42,)),
                "vf_tuned": analysis.ensembles("defectcon", "vf_tuned", (42,))}
    if name == "operating-region":
        return analysis.operating_region()
    if name == "natural-rate":
        return analysis.natural_rate()
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis", choices=["paired", "paired-vf", "variance", "per-seed", "ensembles",
                                         "operating-region", "natural-rate"])
    ap.add_argument("--out", type=Path, default=results.METRICS_ROOT)
    a = ap.parse_args()
    out = a.out / f"{a.analysis.replace('-', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_json(compute(a.analysis)), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
