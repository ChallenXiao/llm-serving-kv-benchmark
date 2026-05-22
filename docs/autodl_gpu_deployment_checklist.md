
# AutoDL / 4090 GPU Deployment Checklist

This document records the cloud deployment procedure for running real vLLM and SGLang serving benchmarks.

---

## 1. Recommended GPU

**Recommended single-GPU options:**
- RTX 4090 24GB
- RTX 3090 24GB
- A10 24GB

**Default benchmark model:** `Qwen/Qwen2.5-7B-Instruct`

**Recommended max model length for initial tests:** `4096`

---

## 2. Check GPU Environment

```bash
nvidia-smi
```

> **Expected Output:**
> 
> - CUDA-visible NVIDIA GPU
>     
> - GPU memory around 24GB for 4090 / 3090 / A10
>     
> - Driver available
>     

## 3. Clone Project

**Via SSH:**

```bash
git clone git@github.com:ChallenXiao/llm-serving-kv-benchmark.git
cd llm-serving-kv-benchmark
```

**Via HTTPS (If SSH is not configured):**

```bash
git clone https://github.com/ChallenXiao/llm-serving-kv-benchmark.git
cd llm-serving-kv-benchmark
```

## 4. Hugging Face Login

If the model requires authentication, log in via CLI:

```bash
huggingface-cli login
```

Or set the token directly in your environment:

```bash
export HF_TOKEN=your_token_here
```

## 5. Option A: Launch vLLM with Docker

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct \
PORT=8000 \
MAX_MODEL_LEN=4096 \
GPU_MEM_UTIL=0.90 \
PREFIX_CACHE=1 \
bash launch/launch_vllm.sh
```

**Health check:**

```bash
curl http://127.0.0.1:8000/v1/models
```

## 6. Option B: Launch SGLang with Docker

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct \
PORT=30000 \
MAX_MODEL_LEN=4096 \
MEM_FRACTION=0.85 \
RADIX_CACHE=1 \
bash launch/launch_sglang.sh
```

**Health check:**

```bash
curl http://127.0.0.1:30000/v1/models
```

## 7. Run Custom Benchmark: vLLM

### Short prompt

```bash
python -m benchmark.run_benchmark \
  --engine vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --workload short \
  --concurrency 1 \
  --num-requests 50 \
  --input-tokens 128 \
  --max-tokens 128 \
  --output results/raw/vllm_short_c1.jsonl
```

### Long prompt

```bash
python -m benchmark.run_benchmark \
  --engine vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --workload long \
  --concurrency 1 \
  --num-requests 50 \
  --input-tokens 1024 \
  --max-tokens 128 \
  --output results/raw/vllm_long_c1.jsonl
```

### Shared prefix

```bash
python -m benchmark.run_benchmark \
  --engine vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --workload shared_prefix \
  --concurrency 1 \
  --num-requests 50 \
  --shared-prefix-tokens 1024 \
  --max-tokens 128 \
  --output results/raw/vllm_shared_prefix_c1.jsonl
```

## 8. Run Custom Benchmark: SGLang

### Short prompt

```bash
python -m benchmark.run_benchmark \
  --engine sglang \
  --base-url http://127.0.0.1:30000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --workload short \
  --concurrency 1 \
  --num-requests 50 \
  --input-tokens 128 \
  --max-tokens 128 \
  --output results/raw/sglang_short_c1.jsonl
```

### Long prompt

```bash
python -m benchmark.run_benchmark \
  --engine sglang \
  --base-url http://127.0.0.1:30000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --workload long \
  --concurrency 1 \
  --num-requests 50 \
  --input-tokens 1024 \
  --max-tokens 128 \
  --output results/raw/sglang_long_c1.jsonl
```

### Shared prefix

```bash
python -m benchmark.run_benchmark \
  --engine sglang \
  --base-url http://127.0.0.1:30000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --workload shared_prefix \
  --concurrency 1 \
  --num-requests 50 \
  --shared-prefix-tokens 1024 \
  --max-tokens 128 \
  --output results/raw/sglang_shared_prefix_c1.jsonl
```

## 9. Recommended Concurrency Sweep

### For short prompt (vLLM)

```bash
for c in 1 4 8 16 32
do
  python -m benchmark.run_benchmark \
    --engine vllm \
    --base-url http://127.0.0.1:8000/v1 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --workload short \
    --concurrency $c \
    --num-requests 100 \
    --input-tokens 128 \
    --max-tokens 128 \
    --output results/raw/vllm_short_c${c}.jsonl
done
```

### For long prompt (vLLM)

```bash
for c in 1 4 8
do
  python -m benchmark.run_benchmark \
    --engine vllm \
    --base-url http://127.0.0.1:8000/v1 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --workload long \
    --concurrency $c \
    --num-requests 50 \
    --input-tokens 1024 \
    --max-tokens 128 \
    --output results/raw/vllm_long_c${c}.jsonl
done
```

> **Note:** Repeat the same commands for SGLang by changing:
> 
> - `--engine sglang`
>     
> - `--base-url http://127.0.0.1:30000/v1`
>     
> - `--output results/raw/sglang_...`
>     

## 10. Generate Summary and Plots

```bash
python -m benchmark.plot_results \
  --raw-dir results/raw \
  --summary-output results/processed/summary.csv \
  --plot-dir plots
```

## 11. Files to Save Before Releasing GPU Instance

Before shutting down the cloud GPU instance, make sure to download or push the following artifacts:

- `results/raw/*.jsonl`
    
- `results/processed/summary.csv`
    
- `plots/*.png`
    
**Recommended check:**
```bash
git status
```