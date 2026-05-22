#!/usr/bin/env bash
set -e

MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
PREFIX_CACHE=${PREFIX_CACHE:-1}

PREFIX_FLAG="--enable-prefix-caching"
if [ "$PREFIX_CACHE" = "0" ]; then
  PREFIX_FLAG="--no-enable-prefix-caching"
fi

echo "Launching vLLM..."
echo "MODEL=${MODEL}"
echo "PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "PREFIX_CACHE=${PREFIX_CACHE}"

docker run --rm \
  --runtime nvidia \
  --gpus all \
  -p ${PORT}:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=${HF_TOKEN} \
  vllm/vllm-openai:latest \
  --model ${MODEL} \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len ${MAX_MODEL_LEN} \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  ${PREFIX_FLAG}
