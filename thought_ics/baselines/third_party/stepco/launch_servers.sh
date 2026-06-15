#!/usr/bin/env bash
# Launch the two vLLM OpenAI-compatible servers required for StepCo eval.
#
# GPU layout (3 GPUs total):
#   - GPUs 0,1: Math-Shepherd verifier (TP=2)  on port 8002
#   - GPU  2  : base reasoner          (TP=1)  on port 8001
#
# Override defaults via env vars:
#   BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct \
#   VERIFIER_MODEL=peiyi9979/math-shepherd-mistral-7b-prm \
#   BASE_GPU=2 VERIFIER_GPUS=0,1 \
#   BASE_PORT=8001 VERIFIER_PORT=8002 \
#   ./launch_servers.sh
#
# After both servers are up (check the printed log files), run:
#   python 3p_baselines/stepco/eval_stepco.py \
#     --base-model-url http://localhost:${BASE_PORT:-8001} \
#     --base-model-name "${BASE_MODEL}" \
#     --verifier-url http://localhost:${VERIFIER_PORT:-8002} \
#     --model llama8b --dataset math500 --n-problems 100

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
VERIFIER_MODEL="${VERIFIER_MODEL:-peiyi9979/math-shepherd-mistral-7b-prm}"

BASE_GPU="${BASE_GPU:-2}"
VERIFIER_GPUS="${VERIFIER_GPUS:-0,1}"

BASE_PORT="${BASE_PORT:-8001}"
VERIFIER_PORT="${VERIFIER_PORT:-8002}"

LOG_DIR="${LOG_DIR:-/tmp/stepco_servers}"
mkdir -p "$LOG_DIR"

BASE_LOG="$LOG_DIR/base_${BASE_PORT}.log"
VERIFIER_LOG="$LOG_DIR/verifier_${VERIFIER_PORT}.log"

echo "[launch_servers] Starting Math-Shepherd verifier on GPUs ${VERIFIER_GPUS} (port ${VERIFIER_PORT})"
CUDA_VISIBLE_DEVICES="${VERIFIER_GPUS}" \
  python -m vllm.entrypoints.openai.api_server \
    --model "${VERIFIER_MODEL}" \
    --port "${VERIFIER_PORT}" \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    > "$VERIFIER_LOG" 2>&1 &
VERIFIER_PID=$!
echo "[launch_servers]   verifier pid=${VERIFIER_PID}, log=${VERIFIER_LOG}"

echo "[launch_servers] Starting base reasoner on GPU ${BASE_GPU} (port ${BASE_PORT})"
CUDA_VISIBLE_DEVICES="${BASE_GPU}" \
  python -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL}" \
    --port "${BASE_PORT}" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    > "$BASE_LOG" 2>&1 &
BASE_PID=$!
echo "[launch_servers]   base     pid=${BASE_PID}, log=${BASE_LOG}"

echo "[launch_servers] Both servers started in background."
echo "[launch_servers] Tail logs to monitor:"
echo "  tail -f ${BASE_LOG}"
echo "  tail -f ${VERIFIER_LOG}"
echo
echo "[launch_servers] Wait for both to print 'Uvicorn running on ...' before launching eval."
echo "[launch_servers] To kill: kill ${VERIFIER_PID} ${BASE_PID}"
echo
echo "VERIFIER_PID=${VERIFIER_PID}"
echo "BASE_PID=${BASE_PID}"
