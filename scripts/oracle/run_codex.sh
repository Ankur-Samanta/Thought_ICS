#!/bin/bash
# Run Codex as an oracle judge on every batch, with a model of your choice.
#   scripts/oracle/run_codex.sh <codex-model> [oracle_dir]
# Writes <oracle_dir>/labels_codex-<model>_<k>.json, then validates them.
set -euo pipefail
MODEL=${1:?codex model, e.g. gpt-5}
ODIR=${2:-results/oracle/part1}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUDGE="codex-${MODEL//_/-}"
for b in "$ODIR"/batch_*.json; do
    k=$(basename "$b" .json | cut -d_ -f2)
    out="$ODIR/labels_${JUDGE}_${k}.json"
    [ -s "$out" ] && { echo "skip $out (exists)"; continue; }
    echo "== $JUDGE batch $k"
    codex exec --model "$MODEL" "$(python3 "$HERE/task.py" "$ODIR" "$JUDGE" "$k")"
done
python3 "$HERE/validate.py" "$ODIR"
