#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}

echo "========== Prefix Cache / RadixAttention Ablation =========="

echo
echo "1. vLLM prefix cache ON"
ENGINE_NAME=vllm_prefix_on \
PREFIX_CACHE=1 \
MODEL="${MODEL}" \
bash scripts/run_4090_vllm_experiment.sh

echo
echo "2. vLLM prefix cache OFF"
ENGINE_NAME=vllm_prefix_off \
PREFIX_CACHE=0 \
MODEL="${MODEL}" \
bash scripts/run_4090_vllm_experiment.sh

echo
echo "3. SGLang RadixAttention ON"
ENGINE_NAME=sglang_radix_on \
RADIX_CACHE=1 \
MODEL="${MODEL}" \
bash scripts/run_4090_sglang_experiment.sh

echo
echo "4. SGLang RadixAttention OFF"
ENGINE_NAME=sglang_radix_off \
RADIX_CACHE=0 \
MODEL="${MODEL}" \
bash scripts/run_4090_sglang_experiment.sh

echo
echo "All ablation experiments completed."
echo "Check experiments/ directory."
