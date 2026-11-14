#!/bin/bash
# Submit the paper's runs to SLURM, one job per (model, benchmark).
#
#   scripts/submit.sh <experiment> <model|all> <benchmark|all> [extra args for the module]
#
# experiments (results go to $OUT_DIR, default ./results):
#   oracle        Thought-ICS + Token-ICS under oracle verification          (thought_ics.ics)
#   self          Thought-ICS + Token-ICS under self-verification            (thought_ics.ics --verify self)
#   self-fb-all   Thought-ICS-F: verification and localization feedback      (needs the `self` runs)
#   self-fb-loc   ablation: verification feedback to the localizer only      (needs the `self` runs)
#   baselines     Self-Refine and CoVe                                       (needs the `self` runs)
#   mv            majority vote over 10 samples (MV@10)
#
# Experiments that start from the `self` runs' first attempts wait (afterok) for a queued or running
# `self` job of the same model and benchmark, and otherwise use the latest finished `self` run.
# Smoke test: scripts/submit.sh self gptoss20b aime --limit 5 --max-ics-iterations 2
#
# Cluster settings can be overridden through the environment: PARTITION, CONDA_PATH, CONDA_ENV,
# TIME, OUT_DIR, DRY_RUN=1 (print the jobs instead of submitting them).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARTITION="${PARTITION:-a100}"
CONDA_PATH="${CONDA_PATH:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-thought-ics}"
TIME="${TIME:-48:00:00}"
OUT_DIR="${OUT_DIR:-$REPO/results}"
LOG_DIR="$OUT_DIR/logs"
DRY_RUN="${DRY_RUN:-0}"

MODELS=(llama3b qwen7b llama8b qwen14b gptoss20b qwen32b llama70b gptoss120b)
BENCHMARKS=(amc23 aime math500 csqa gpqa mathqa)
declare -A GPUS=([llama3b]=1 [qwen7b]=1 [llama8b]=1 [qwen14b]=2 [gptoss20b]=1 [qwen32b]=4 [llama70b]=8 [gptoss120b]=4)
declare -A RUN_TAG=([math500]=math500_L5)

experiment=${1:?experiment}; model=${2:?model or all}; bench=${3:?benchmark or all}; shift 3
case "$experiment" in
    oracle)      module=thought_ics.ics;           args="--verify oracle";;
    self)        module=thought_ics.ics;           args="--verify self";;
    self-fb-all) module=thought_ics.ics;           args="--verify self --feedback all --method thought";;
    self-fb-loc) module=thought_ics.ics;           args="--verify self --feedback loc --method thought";;
    baselines)   module=thought_ics.baselines;     args="";;
    mv)          module=thought_ics.majority_vote; args="";;
    *) sed -n '2,20p' "$0"; exit 1;;
esac
needs_self=0
[[ "$experiment" == self-fb-* || "$experiment" == baselines ]] && needs_self=1

models=("$model"); [ "$model" = all ] && models=("${MODELS[@]}")
benches=("$bench"); [ "$bench" = all ] && benches=("${BENCHMARKS[@]}")
mkdir -p "$LOG_DIR"

for m in "${models[@]}"; do
    [ -n "${GPUS[$m]:-}" ] || { echo "unknown model $m"; exit 1; }
    for b in "${benches[@]}"; do
        name="ics_${experiment}_${m}_${b}"
        extra="" dep=""
        if [ $needs_self = 1 ]; then
            jid=$(squeue -u "$USER" -h -n "ics_self_${m}_${b}" -o %i 2>/dev/null | head -1 || true)
            if [ -n "$jid" ]; then
                if [ "$experiment" = baselines ]; then
                    dep="--dependency=afterok:${jid}"   # baselines find the finished run themselves
                else
                    echo "self run for $m $b still in the queue (job $jid); submit $experiment once it has finished"; continue
                fi
            else
                tag=${RUN_TAG[$b]:-$b}
                init=$(ls -d "$OUT_DIR/${m}_${tag}_self_2"*/ 2>/dev/null | tail -1 || true)
                [ -n "$init" ] || { echo "no self-verified run for $m $b in $OUT_DIR (run 'self' first)"; exit 1; }
                extra="--init-from ${init%/}"
            fi
        fi
        cmd="python -m $module --model $m --dataset $b $args $extra --out-dir $OUT_DIR $*"
        if [ "$DRY_RUN" = 1 ]; then echo "[${GPUS[$m]} GPU] $name: $cmd"; continue; fi
        sbatch $dep --partition="$PARTITION" --nodes=1 --gpus-per-node="${GPUS[$m]}" --cpus-per-gpu=10 \
               --time="$TIME" --job-name="$name" \
               --output="$LOG_DIR/${name}_%j.out" --error="$LOG_DIR/${name}_%j.err" \
               --wrap="bash -c 'source $CONDA_PATH/etc/profile.d/conda.sh && conda activate $CONDA_ENV && cd $REPO && export VLLM_CACHE_ROOT=/tmp/vllm_cache_\$SLURM_JOB_ID && $cmd'"
    done
done
