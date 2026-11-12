#!/bin/bash
# Oracle-localization pipeline (clean vs erroneous prefixes, both methods).
#   scripts/budget_study/oracle/pipeline.sh build     # sample wrong first attempts -> batches
#   scripts/budget_study/oracle/pipeline.sh task <judge> <k>   # print task text for any judge
#   scripts/budget_study/oracle/pipeline.sh status    # which judges/batches are done + valid
#   scripts/budget_study/oracle/pipeline.sh analyze   # consensus + clean/erroneous analysis
# Judges write labels_<judge>_<k>.json; any number of judges, any model:
#   Codex:   scripts/budget_study/oracle/run_codex.sh <model>
#   Claude:  subagents given the output of `pipeline.sh task <name> <k>`
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
FULL=${FULL:-$REPO/experiments_budget_study/full}
ODIR=${ODIR:-$REPO/experiments_budget_study/oracle}
PY=${PY:-/home/$USER/miniconda3/envs/scpo/bin/python}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
case "${1:-}" in
    build)   "$PY" "$HERE/build.py" "$FULL" "$ODIR" "${2:-30}" "${3:-3}" ;;
    task)    python3 "$HERE/task.py" "$ODIR" "${2:?judge}" "${3:?batch}" ;;
    status)  python3 "$HERE/validate.py" "$ODIR" || true ;;
    analyze) python3 "$HERE/validate.py" "$ODIR" && python3 "$HERE/analyze.py" "$FULL" "$ODIR" ;;
    *) sed -n '2,10p' "$0"; exit 1 ;;
esac
