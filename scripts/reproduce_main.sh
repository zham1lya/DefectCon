#!/usr/bin/env bash
# Tables 3-6 and Fig. 3 (main comparison, no VF) from the per-image scores in
# results/scores (or $DEFECTCON_RESULTS/scores). CPU only, about a minute.
#
#     bash scripts/reproduce_main.sh [output directory]
#
# Default output: results/metrics/ for the tables, figures/ for Fig. 3.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-}
METRICS=(); FIGURES=()
if [[ -n "$OUT" ]]; then METRICS=(--out "$OUT"); FIGURES=(--out "$OUT"); fi

python scripts/aggregate.py --condition main "${METRICS[@]}"   # Table 3
python scripts/analyze.py paired "${METRICS[@]}"               # Table 4
python scripts/analyze.py variance "${METRICS[@]}"             # Table 5
python scripts/analyze.py per-seed "${METRICS[@]}"             # Section 5.2.2
python scripts/analyze.py ensembles "${METRICS[@]}"            # Section 5.2.3
python scripts/analyze.py operating-region "${METRICS[@]}"     # Table 6
python scripts/figures.py operating-region "${FIGURES[@]}"     # Fig. 3
