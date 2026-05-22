#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
PORT=${PORT:-30000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MEM_FRACTION=${MEM_FRACTION:-0.90}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-64}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-16384}
RADIX_CACHE=${RADIX_CACHE:-1}

ENGINE_NAME=${ENGINE_NAME:-"sglang_radix_on"}
CONTAINER_NAME=${CONTAINER_NAME:-"sglang_bench"}

RADIX_FLAG=""
if [ "$RADIX_CACHE" = "0" ]; then
  RADIX_FLAG="--disable-radix-cache"
  ENGINE_NAME=${ENGINE_NAME:-"sglang_radix_off"}
fi

echo "========== Starting SGLang Experiment =========="
echo "MODEL=${MODEL}"
echo "PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "MEM_FRACTION=${MEM_FRACTION}"
echo "MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}"
echo "MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS}"
echo "RADIX_CACHE=${RADIX_CACHE}"
echo "ENGINE_NAME=${ENGINE_NAME}"

mkdir -p server_logs

echo
echo "Removing old container if exists..."
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo
echo "Launching SGLang container..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --shm-size 32g \
  -p ${PORT}:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=${HF_TOKEN:-} \
  --ipc=host \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port 30000 \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
    ${RADIX_FLAG}

echo
echo "Waiting for SGLang server..."
for i in $(seq 1 120); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "SGLang server is ready."
    break
  fi

  if [ "$i" -eq 120 ]; then
    echo "ERROR: SGLang server did not become ready."
    docker logs "${CONTAINER_NAME}" | tail -200
    exit 1
  fi

  sleep 5
done

echo
echo "Running benchmark sweep..."
ENGINE="${ENGINE_NAME}" \
BASE_URL="http://127.0.0.1:${PORT}/v1" \
MODEL="${MODEL}" \
bash scripts/run_cloud_sweep.sh

LATEST_EXP=$(ls -td experiments/${ENGINE_NAME}_* | head -1)

echo
echo "Saving SGLang server logs..."
docker logs "${CONTAINER_NAME}" > "${LATEST_EXP}/logs/sglang_server.log" 2>&1 || true
nvidia-smi > "${LATEST_EXP}/logs/nvidia_smi_after.log" 2>&1 || true

echo
echo "Stopping SGLang container..."
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo
echo "SGLang experiment completed."
echo "Experiment directory: ${LATEST_EXP}"
