#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
PREFIX_CACHE=${PREFIX_CACHE:-1}

CONTAINER_NAME=${CONTAINER_NAME:-"vllm_bench"}

if [ "$PREFIX_CACHE" = "0" ]; then
  PREFIX_FLAG="--no-enable-prefix-caching"
  ENGINE_NAME=${ENGINE_NAME:-"vllm_prefix_off"}
else
  PREFIX_FLAG="--enable-prefix-caching"
  ENGINE_NAME=${ENGINE_NAME:-"vllm_prefix_on"}
fi

echo "========== Starting vLLM Experiment =========="
echo "MODEL=${MODEL}"
echo "PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "GPU_MEM_UTIL=${GPU_MEM_UTIL}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "PREFIX_CACHE=${PREFIX_CACHE}"
echo "ENGINE_NAME=${ENGINE_NAME}"

mkdir -p server_logs

echo
echo "Removing old container if exists..."
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo
echo "Launching vLLM container..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  --runtime nvidia \
  --gpus all \
  -p ${PORT}:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=${HF_TOKEN:-} \
  vllm/vllm-openai:latest \
  --model "${MODEL}" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  ${PREFIX_FLAG}

echo
echo "Waiting for vLLM server..."
for i in $(seq 1 120); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM server is ready."
    break
  fi

  if [ "$i" -eq 120 ]; then
    echo "ERROR: vLLM server did not become ready."
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
echo "Saving vLLM server logs..."
docker logs "${CONTAINER_NAME}" > "${LATEST_EXP}/logs/vllm_server.log" 2>&1 || true
nvidia-smi > "${LATEST_EXP}/logs/nvidia_smi_after.log" 2>&1 || true

echo
echo "Stopping vLLM container..."
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo
echo "vLLM experiment completed."
echo "Experiment directory: ${LATEST_EXP}"
