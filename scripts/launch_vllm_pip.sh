#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
PREFIX_CACHE=${PREFIX_CACHE:-1}

if [ "$PREFIX_CACHE" = "0" ]; then
  PREFIX_FLAG="--no-enable-prefix-caching"
else
  PREFIX_FLAG="--enable-prefix-caching"
fi

echo "========== Launching vLLM without Docker =========="
echo "MODEL=${MODEL}"
echo "PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "GPU_MEM_UTIL=${GPU_MEM_UTIL}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "PREFIX_CACHE=${PREFIX_CACHE}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype auto \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  ${PREFIX_FLAG}
