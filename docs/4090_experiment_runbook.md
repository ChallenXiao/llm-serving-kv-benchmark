# RTX 4090 Cloud Experiment Runbook

This document explains how to run the real GPU experiments for the LLM serving benchmark project.

## 1. Purpose

The local MacBook setup is used to validate the benchmark pipeline with a mock OpenAI-compatible server.

The RTX 4090 server is used to run real vLLM / SGLang serving experiments with Qwen/Qwen2.5-7B-Instruct.

The real experiments measure:

- request throughput
- output tokens per second
- TTFT
- TPOT
- P95 / P99 end-to-end latency
- behavior under short prompt, long prompt, and shared-prefix workloads
- prefix cache / RadixAttention ablation effects

## 2. Recommended Cloud Image

Use an Ubuntu + CUDA + Docker + NVIDIA Container Toolkit image if available.

Minimum checks:

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

3. Clone Repository
```
git clone https://github.com/ChallenXiao/llm-serving-kv-benchmark.git
cd llm-serving-kv-benchmark
```
4. Setup Environment
```
bash scripts/setup_4090_env.sh
source .venv/bin/activate
```
5. Hugging Face Token

If needed:
```
export HF_TOKEN=your_huggingface_token
```
6. Conservative First Run

Start with a conservative vLLM run:
```
MAX_MODEL_LEN=4096 \
GPU_MEM_UTIL=0.90 \
MAX_NUM_SEQS=32 \
MAX_NUM_BATCHED_TOKENS=8192 \
MODEL=Qwen/Qwen2.5-7B-Instruct \
bash scripts/run_4090_vllm_experiment.sh
```
If this works, try the stronger 4090 configuration:
```
MAX_MODEL_LEN=8192 \
GPU_MEM_UTIL=0.92 \
MAX_NUM_SEQS=64 \
MAX_NUM_BATCHED_TOKENS=16384 \
MODEL=Qwen/Qwen2.5-7B-Instruct \
bash scripts/run_4090_vllm_experiment.sh
```
7. Run SGLang

Conservative:
```
MAX_MODEL_LEN=4096 \
MEM_FRACTION=0.85 \
MAX_RUNNING_REQUESTS=32 \
MAX_PREFILL_TOKENS=8192 \
MODEL=Qwen/Qwen2.5-7B-Instruct \
bash scripts/run_4090_sglang_experiment.sh
```

Stronger:
```
MAX_MODEL_LEN=8192 \
MEM_FRACTION=0.90 \
MAX_RUNNING_REQUESTS=64 \
MAX_PREFILL_TOKENS=16384 \
MODEL=Qwen/Qwen2.5-7B-Instruct \
bash scripts/run_4090_sglang_experiment.sh
```
8. Run Prefix Cache Ablation
```
MODEL=Qwen/Qwen2.5-7B-Instruct \
bash scripts/run_prefix_cache_ablation.sh
```

This runs:

vLLM prefix cache on
vLLM prefix cache off
SGLang RadixAttention on
SGLang RadixAttention off
9. If GPU Memory Is Not Fully Used

Increase pressure in this order:

Increase concurrency
Increase output tokens
Increase input tokens
Increase max model length
Increase max number of sequences
Increase max batched tokens
Increase GPU memory utilization / memory fraction

Example:
```
SHORT_CONCURRENCIES="1 4 8 16 32 64 96" \
LONG_CONCURRENCIES="1 2 4 8 16 24" \
SHARED_CONCURRENCIES="1 4 8 16 32 64" \
SHORT_MAX_TOKENS=384 \
LONG_INPUT_TOKENS=3072 \
LONG_MAX_TOKENS=384 \
SHARED_PREFIX_TOKENS=3072 \
SHARED_MAX_TOKENS=384 \
bash scripts/run_4090_vllm_experiment.sh
```
10. If OOM Happens

For vLLM:
```
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.85
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=8192
```
For SGLang:
```
MAX_MODEL_LEN=4096
MEM_FRACTION=0.80
MAX_RUNNING_REQUESTS=32
MAX_PREFILL_TOKENS=8192
```
Also reduce benchmark concurrency.

11. Expected Results

Short prompt:

throughput should increase as concurrency increases
P95 latency should rise at high concurrency

Long prompt:

TTFT should be much higher than short prompt
throughput should be lower than short prompt

Shared prefix:

with prefix cache / RadixAttention enabled, TTFT should improve after cache warmup
without cache, shared_prefix should behave closer to long prompt

12. Mapping to Resume Claims

This experiment supports the following resume claims:

Built OpenAI-compatible LLM serving benchmark for vLLM and SGLang.
Designed short-prompt, long-context, and shared-prefix workloads.
Measured request throughput, output tokens/s, TTFT, TPOT, and P95 latency.
Analyzed prefill/decode bottlenecks under different concurrency levels.
Evaluated prefix caching and RadixAttention under shared-prefix workloads.
Implemented a KV cache allocator simulator to explain PagedAttention-style memory efficiency.