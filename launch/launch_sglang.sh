#!/usr/bin/env bash
set -e

MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
PORT=${PORT:-30000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MEM_FRACTION=${MEM_FRACTION:-0.85}
RADIX_CACHE=${RADIX_CACHE:-1}

RADIX_FLAG=""
if [ "$RADIX_CACHE" = "0" ]; then
  RADIX_FLAG="--disable-radix-cache"
fi

echo "Launching SGLang..."
echo "MODEL=${MODEL}"
echo "PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "RADIX_CACHE=${RADIX_CACHE}"

docker run --rm \
  --gpus all \
  --shm-size 32g \
  -p ${PORT}:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=${HF_TOKEN} \
  --ipc=host \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path ${MODEL} \
    --host 0.0.0.0 \
    --port 30000 \
    --context-length ${MAX_MODEL_LEN} \
    --mem-fraction-static ${MEM_FRACTION} \
    ${RADIX_FLAG}
