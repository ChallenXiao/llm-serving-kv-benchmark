#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
PORT=${PORT:-8000}
PREFIX_CACHE=${PREFIX_CACHE:-1}

if [ "$PREFIX_CACHE" = "0" ]; then
  ENGINE_NAME=${ENGINE_NAME:-"vllm_prefix_off"}
else
  ENGINE_NAME=${ENGINE_NAME:-"vllm_prefix_on"}
fi

mkdir -p server_logs

echo "========== Starting vLLM Server =========="

source .venv-gpu/bin/activate

MODEL="${MODEL}" \
PORT="${PORT}" \
PREFIX_CACHE="${PREFIX_CACHE}" \
bash scripts/launch_vllm_pip.sh > "server_logs/${ENGINE_NAME}.log" 2>&1 &

SERVER_PID=$!

echo "vLLM PID: ${SERVER_PID}"
echo "Waiting for server..."

for i in $(seq 1 120); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM server ready."
    break
  fi

  if [ "$i" -eq 120 ]; then
    echo "ERROR: vLLM server did not become ready."
    tail -200 "server_logs/${ENGINE_NAME}.log"
    kill ${SERVER_PID} || true
    exit 1
  fi

  sleep 5
done

ENGINE="${ENGINE_NAME}" \
BASE_URL="http://127.0.0.1:${PORT}/v1" \
MODEL="${MODEL}" \
bash scripts/run_cloud_sweep.sh

LATEST_EXP=$(ls -td experiments/${ENGINE_NAME}_* | head -1)
cp "server_logs/${ENGINE_NAME}.log" "${LATEST_EXP}/logs/vllm_server.log" || true
nvidia-smi > "${LATEST_EXP}/logs/nvidia_smi_after.log" 2>&1 || true

echo "Stopping vLLM server..."
kill ${SERVER_PID} || true

echo "Done. Experiment: ${LATEST_EXP}"
