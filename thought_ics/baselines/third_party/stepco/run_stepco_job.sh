#!/usr/bin/env bash
# Inner runner for a single StepCo SLURM job.
#
# Lifecycle:
#   1. Launch the Math-Shepherd verifier vLLM server (background)
#   2. Launch the base reasoner vLLM server (background)
#   3. Wait for both /v1/models endpoints to come up
#   4. Run eval_stepco.py
#   5. Tear down both servers (always, via trap)
#
# Args (positional):
#   $1  MODEL_NICKNAME   short name (e.g. llama8b)        — used for output dirs
#   $2  BASE_HF          HF model id of the base reasoner
#   $3  DATASET          dataset name (e.g. math500)
#   $4  N_PROBLEMS       number of problems to evaluate
#   $5  BASE_GPUS        comma-separated GPU ids for base reasoner (e.g. "0")
#   $6  BASE_TP          tensor-parallel size for base reasoner
#   $7  VERIFIER_GPUS    comma-separated GPU ids for verifier (e.g. "1,2")
#   $8  LEVEL            optional MATH-500 level filter (pass empty string if N/A)
#
# Ports: defaults to 8001 (base) / 8002 (verifier). Override via env vars
# BASE_PORT / VERIFIER_PORT. Inside SLURM each job typically gets an exclusive
# node so fixed ports are fine; the script also derives a per-job offset from
# SLURM_JOB_ID as a safety net if ports happen to be in use.
#
# Required env: TREE_DIR (path to repo root), CONDA env activated by caller.

set -euo pipefail

MODEL_NICKNAME="${1:?model nickname required}"
BASE_HF="${2:?base HF model required}"
DATASET="${3:?dataset required}"
N_PROBLEMS="${4:?n_problems required}"
BASE_GPUS="${5:?base GPUs required}"
BASE_TP="${6:?base TP size required}"
VERIFIER_GPUS="${7:?verifier GPUs required}"
LEVEL="${8:-}"

TREE_DIR="${TREE_DIR:-$(pwd)}"
VERIFIER_HF="${VERIFIER_HF:-peiyi9979/math-shepherd-mistral-7b-prm}"

# Derive verifier TP from GPU count (e.g. "1" -> 1, "1,2" -> 2)
VERIFIER_TP=$(echo "$VERIFIER_GPUS" | awk -F',' '{print NF}')

# Map the relative GPU indices passed by submit_stepco.sh (e.g. "0", "1,2")
# into the absolute physical IDs that SLURM actually allocated to this job.
# Without this remap, overriding CUDA_VISIBLE_DEVICES with absolute IDs like
# "0" or "1" can refer to GPUs that aren't in our cgroup if SLURM allocated
# non-contiguous devices (e.g. physical GPUs 3 and 5).
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra _SLURM_GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    map_relative_gpus() {
        local rel=$1
        local out=""
        local r
        IFS=',' read -ra _rel_array <<< "$rel"
        for r in "${_rel_array[@]}"; do
            if [ "$r" -ge "${#_SLURM_GPU_ARRAY[@]}" ]; then
                echo "[run_stepco_job] ERROR: relative GPU index $r out of range (slurm gave ${#_SLURM_GPU_ARRAY[@]} GPUs: $CUDA_VISIBLE_DEVICES)" >&2
                exit 1
            fi
            out="${out:+$out,}${_SLURM_GPU_ARRAY[$r]}"
        done
        echo "$out"
    }
    BASE_GPUS=$(map_relative_gpus "$BASE_GPUS")
    VERIFIER_GPUS=$(map_relative_gpus "$VERIFIER_GPUS")
    echo "[run_stepco_job] SLURM allocated GPUs: $CUDA_VISIBLE_DEVICES"
    echo "[run_stepco_job] After remap: BASE_GPUS=$BASE_GPUS  VERIFIER_GPUS=$VERIFIER_GPUS"
fi

# Per-job port offset (0-99) to reduce collision risk if jobs share a node.
PORT_OFFSET=$(( ${SLURM_JOB_ID:-0} % 100 ))
BASE_PORT="${BASE_PORT:-$((8001 + PORT_OFFSET * 2))}"
VERIFIER_PORT="${VERIFIER_PORT:-$((8002 + PORT_OFFSET * 2))}"

LOG_DIR="${LOG_DIR:-/tmp/stepco_servers_${SLURM_JOB_ID:-$$}}"
mkdir -p "$LOG_DIR"
BASE_LOG="$LOG_DIR/base_${BASE_PORT}.log"
VERIFIER_LOG="$LOG_DIR/verifier_${VERIFIER_PORT}.log"

echo "[run_stepco_job] TREE_DIR=$TREE_DIR"
echo "[run_stepco_job] MODEL=$MODEL_NICKNAME ($BASE_HF)  DATASET=$DATASET  N=$N_PROBLEMS  LEVEL=${LEVEL:-none}"
echo "[run_stepco_job] BASE_GPUS=$BASE_GPUS (TP=$BASE_TP) port=$BASE_PORT"
echo "[run_stepco_job] VERIFIER_GPUS=$VERIFIER_GPUS (TP=$VERIFIER_TP) port=$VERIFIER_PORT"
echo "[run_stepco_job] LOG_DIR=$LOG_DIR"

# ----------------------------------------------------------------------------
# Launch verifier (Math-Shepherd) — TP=2 on the verifier GPUs
# ----------------------------------------------------------------------------
echo "[run_stepco_job] Launching verifier..."
CUDA_VISIBLE_DEVICES="$VERIFIER_GPUS" \
  python -m vllm.entrypoints.openai.api_server \
    --model "$VERIFIER_HF" \
    --port "$VERIFIER_PORT" \
    --tensor-parallel-size "$VERIFIER_TP" \
    --dtype float16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    > "$VERIFIER_LOG" 2>&1 &
VERIFIER_PID=$!
echo "[run_stepco_job]   verifier_pid=$VERIFIER_PID"

# ----------------------------------------------------------------------------
# Launch base reasoner
# ----------------------------------------------------------------------------
echo "[run_stepco_job] Launching base reasoner..."
CUDA_VISIBLE_DEVICES="$BASE_GPUS" \
  python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_HF" \
    --port "$BASE_PORT" \
    --tensor-parallel-size "$BASE_TP" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    > "$BASE_LOG" 2>&1 &
BASE_PID=$!
echo "[run_stepco_job]   base_pid=$BASE_PID"

# ----------------------------------------------------------------------------
# Cleanup trap
# ----------------------------------------------------------------------------
cleanup() {
  echo "[run_stepco_job] Cleanup: tearing down servers..."
  kill -TERM "$VERIFIER_PID" "$BASE_PID" 2>/dev/null || true
  sleep 3
  kill -KILL "$VERIFIER_PID" "$BASE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ----------------------------------------------------------------------------
# Wait for both servers
# ----------------------------------------------------------------------------
wait_for_server() {
  local url=$1
  local name=$2
  local pid=$3
  local timeout="${WAIT_TIMEOUT:-1200}"  # 20 min default (large models load slowly)
  local elapsed=0
  while [ "$elapsed" -lt "$timeout" ]; do
    # If the server process died, bail with its log
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[run_stepco_job] ERROR: $name server (pid=$pid) died during startup"
      return 1
    fi
    if curl -sf -o /dev/null "$url/v1/models" 2>/dev/null; then
      echo "[run_stepco_job] $name ready after ${elapsed}s"
      return 0
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  echo "[run_stepco_job] ERROR: $name not ready after ${timeout}s"
  return 1
}

wait_for_server "http://localhost:$VERIFIER_PORT" verifier "$VERIFIER_PID"
wait_for_server "http://localhost:$BASE_PORT"     base     "$BASE_PID"

# ----------------------------------------------------------------------------
# Run eval
# ----------------------------------------------------------------------------
LEVEL_ARG=""
if [ -n "$LEVEL" ]; then
  LEVEL_ARG="--level $LEVEL"
fi

cd "$TREE_DIR"
echo "[run_stepco_job] Running eval_stepco.py..."
python 3p_baselines/stepco/eval_stepco.py \
  --base-model-url "http://localhost:$BASE_PORT" \
  --base-model-name "$BASE_HF" \
  --verifier-url "http://localhost:$VERIFIER_PORT" \
  --verifier-model-name "$VERIFIER_HF" \
  --model "$MODEL_NICKNAME" \
  --dataset "$DATASET" \
  --n-problems "$N_PROBLEMS" \
  $LEVEL_ARG \
  --generation-temp 0.5 \
  --max-tokens 2048 \
  --top-p 0.9 \
  --top-k 50 \
  --threshold 0.5 \
  --max-iterations 5

echo "[run_stepco_job] eval complete"
