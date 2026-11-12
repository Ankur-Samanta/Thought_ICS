#!/bin/bash
# Submit the budget-controlled Thought-ICS vs Token-ICS study (SRPO protocol + matched
# Token-ICS). Each job runs BOTH methods on one model/dataset from a single model load.
#
# Usage:
#   scripts/budget_study/submit.sh smoke                 # 1 tiny job: gptoss20b, aime, 5 problems, 2 iters
#   scripts/budget_study/submit.sh <model> <dataset>     # one job
#   scripts/budget_study/submit.sh <model> all           # 6 jobs
#   scripts/budget_study/submit.sh all all               # 18 jobs (3 models x 6 datasets)
#   scripts/budget_study/submit.sh list                  # print jobs without submitting
#
# Jobs run with the HF hub offline (weights must already be in ~/.cache/huggingface);
# proxy variables from the submitting shell are cleared on the compute node.
# Extra args after the dataset are forwarded to run_study.py
# (e.g. --response-length 8192).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_PATH="/home/${USER}/miniconda3"
CONDA_ENV="scpo"            # vLLM 0.11.2 (gpt-oss support); same env as SRPO/SCPO
PARTITION="a100"
TIME="48:00:00"
OUT_DIR="${REPO}/experiments_budget_study"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

declare -A GPUS=([qwen32b]=4 [gptoss20b]=2 [gptoss120b]=4)
MODELS=(qwen32b gptoss20b gptoss120b)
DATASETS=(amc23 aime math500 csqa gpqa mathqa)

dataset_args() {
    case "$1" in
        amc23)   echo "--dataset amc23 --n-problems 40" ;;
        math500) echo "--dataset math500 --level 5 --n-problems 100" ;;
        *)       echo "--dataset $1 --n-problems 100" ;;
    esac
}

DRY_RUN=false

submit() {  # model dataset_args job_name time extra...
    local model=$1 dargs=$2 name=$3 time=$4; shift 4
    local gpus=${GPUS[$model]}
    local cmd="python -m thought_ics.budget_study.run_study --model ${model} ${dargs} --method both --autonomy-level 2 --out-dir ${OUT_DIR} $*"
    if $DRY_RUN; then echo "[${gpus} GPU] ${name}: ${cmd}"; return; fi
    sbatch --partition=${PARTITION} --nodes=1 --gpus-per-node=${gpus} --cpus-per-gpu=10 \
           --time=${time} --job-name=${name} \
           --output=${LOG_DIR}/${name}_%j.out --error=${LOG_DIR}/${name}_%j.err \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${REPO} && unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy && export HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-1} VLLM_CACHE_ROOT=/tmp/vllm_cache_\$SLURM_JOB_ID && ${cmd}'"
}

case "${1:-}" in
    smoke)
        shift
        submit gptoss20b "--dataset aime --n-problems 5" bs_smoke_gptoss20b_aime 02:00:00 --max-ics-iterations 2 "$@"
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
            for d in "${dsets[@]}"; do submit "$m" "$(dataset_args "$d")" "bs_${m}_${d}" "$TIME" "$@"; done
        done
        ;;
esac
