#!/bin/bash
# Submit the budget-controlled Thought-ICS vs Token-ICS study (SRPO protocol + matched
# Token-ICS). Each job runs BOTH methods on one model/dataset from a single model load.
#
# Usage:
#   scripts/budget_study/submit.sh smoke [model]         # 1 tiny job: aime, 5 problems, 2 iters
#   scripts/budget_study/submit.sh <model> <dataset>     # one job
#   scripts/budget_study/submit.sh <model> all           # 6 jobs
#   scripts/budget_study/submit.sh all all               # 18 jobs (3 models x 6 datasets)
#   scripts/budget_study/submit.sh list                  # print jobs without submitting
#   scripts/budget_study/submit.sh token_shared          # Token-ICS only, starting from each full
#                                                        # run's Thought-ICS initial chains (joined)
#
# Jobs run with the HF hub offline (weights must already be in ~/.cache/huggingface);
# proxy variables from the submitting shell are cleared on the compute node.
# Extra args after the dataset are forwarded to run_study.py
# (e.g. --shared-init --self-verify). Default is --oracle-verify.
# MODELS_OVERRIDE="a b" picks models for "all"; JOB_PREFIX sets the job-name prefix (default bs).
# MODULE=thought_ics.budget_study.baselines runs Self-Refine + CoVe instead; DEPEND_ON_PREFIX=bsSV makes
# each job wait (afterok) for the queued/running job <prefix>_<model>_<dataset>, e.g. the self-verify
# run whose first attempts it starts from (if that job already finished, no dependency is added).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_PATH="/home/${USER}/miniconda3"
CONDA_ENV="scpo"            # vLLM 0.11.2 (gpt-oss support); same env as SRPO/SCPO
PARTITION="a100"
TIME="48:00:00"
OUT_DIR="${REPO}/experiments_budget_study"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

declare -A GPUS=([qwen32b]=4 [gptoss20b]=1 [gptoss120b]=4 [llama3b]=1 [llama8b]=1 [llama70b]=8 [qwen7b]=1 [qwen14b]=2)  # gpt-oss: prefix caching off (see run_study.MODELS)
MODELS=(${MODELS_OVERRIDE:-qwen32b gptoss20b gptoss120b})
DATASETS=(amc23 aime math500 csqa gpqa mathqa)

dataset_args() {
    case "$1" in
        amc23)   echo "--dataset amc23 --n-problems 40" ;;
        math500) echo "--dataset math500 --level 5 --n-problems 100" ;;
        *)       echo "--dataset $1 --n-problems 100" ;;
    esac
}

DRY_RUN=false

RUN_SUBDIR="full"   # full runs -> experiments_budget_study/full, smoke runs -> .../dev

submit() {  # model dataset_args job_name time extra...
    local model=$1 dargs=$2 name=$3 time=$4; shift 4
    local gpus=${GPUS[$model]}
    local method="both"; [[ "$*" == *--token-init-from* ]] && method="token"
    local cmd="python -m ${MODULE:-thought_ics.budget_study.run_study} --model ${model} ${dargs} --method ${method} --out-dir ${OUT_DIR}/${RUN_SUBDIR} $*"
    if $DRY_RUN; then echo "[${gpus} GPU] ${name}: ${cmd}"; return; fi
    local dep=""; [ -n "${DEPEND_ON_PREFIX:-}" ] && { local jid; jid=$(squeue -u "$USER" -h -n "${DEPEND_ON_PREFIX}_${name#*_}" -o %i | head -1); [ -n "$jid" ] && dep="--dependency=afterok:${jid}"; }
    sbatch ${dep} --partition=${PARTITION} --nodes=1 --gpus-per-node=${gpus} --cpus-per-gpu=10 \
           --time=${time} --job-name=${name} \
           --output=${LOG_DIR}/${name}_%j.out --error=${LOG_DIR}/${name}_%j.err \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${REPO} && unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy && export HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-1} VLLM_CACHE_ROOT=/tmp/vllm_cache_\$SLURM_JOB_ID && ${cmd}'"
}

case "${1:-}" in
    smoke)  # smoke [model] : 5 AIME problems, 2 ICS iterations
        shift; m=${1:-gptoss20b}; [ $# -gt 0 ] && shift
        RUN_SUBDIR="dev"
        submit "$m" "--dataset aime --n-problems 5" "bs_smoke_${m}_aime" 02:00:00 --max-ics-iterations 2 "$@"
        ;;
    token_shared|token_shared_list)  # Token-ICS only, initial solutions = joined Thought-ICS chains
        [ "$1" = token_shared_list ] && DRY_RUN=true
        shift
        for d in "${OUT_DIR}"/full/*_L2_2*/; do
            [[ "$d" == *_sharedinit_* || "$d" == *_sharedprompt_* ]] && continue
            [ -f "$d/thought_results.jsonl" ] || continue
            read -r m dargs <<< "$(python3 -c "
import json; c=json.load(open('$d/config.json'))
a=f\"--dataset {c['dataset']} --n-problems {c['n_problems']}\" + (f\" --level {c['level']}\" if c.get('level') else '')
print(c['model'], a)")"
            name="bs_tokshared_${m}_$(basename "$d" | cut -d_ -f2)"
            submit "$m" "$dargs" "$name" 12:00:00 --token-init-from "$d" "$@"
        done
        ;;
    list)
        DRY_RUN=true
        for m in "${MODELS[@]}"; do for d in "${DATASETS[@]}"; do submit "$m" "$(dataset_args "$d")" "bs_${m}_${d}" "$TIME"; done; done
        ;;
    *)
        model=${1:?model or all}; ds=${2:?dataset or all}; shift 2
        models=("$model"); [ "$model" = all ] && models=("${MODELS[@]}")
        dsets=("$ds");     [ "$ds" = all ] && dsets=("${DATASETS[@]}")
        for m in "${models[@]}"; do
            [ -n "${GPUS[$m]:-}" ] || { echo "unknown model $m"; exit 1; }
            for d in "${dsets[@]}"; do submit "$m" "$(dataset_args "$d")" "${JOB_PREFIX:-bs}_${m}_${d}" "$TIME" "$@"; done
        done
        ;;
esac
