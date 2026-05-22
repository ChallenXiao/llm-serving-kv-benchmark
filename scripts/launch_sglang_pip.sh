#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-30000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MEM_FRACTION=${MEM_FRACTION:-0.85}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-32}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
RADIX_CACHE=${RADIX_CACHE:-1}

if [ "$RADIX_CACHE" = "0" ]; then
  RADIX_FLAG="--disable-radix-cache"
else
  RADIX_FLAG=""
fi

echo "========== Launching SGLang without Docker =========="
echo "MODEL=${MODEL}"
echo "PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "MEM_FRACTION=${MEM_FRACTION}"
echo "MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}"
echo "MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS}"
echo "RADIX_CACHE=${RADIX_CACHE}"

python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --context-length "${MAX_MODEL_LEN}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
  ${RADIX_FLAG}
