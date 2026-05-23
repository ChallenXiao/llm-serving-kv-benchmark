# RTX 3090 No-Docker Cloud Deployment Runbook

This document records the real cloud deployment process for the LLM serving benchmark project on an RTX 3090 24GB instance without Docker support.

The goal is to document not only the final commands, but also the engineering adjustments made during cloud deployment: SSH/Git setup, Python environments, CUDA/vLLM compatibility, Hugging Face cache migration, disk cleanup, vLLM serving, benchmark execution, and the next planned SGLang experiments.

---

## 1. Cloud Machine Situation

The rented cloud machine provides:

- GPU: NVIDIA GeForce RTX 3090 24GB
- Driver: CUDA-compatible NVIDIA driver
- Base image: PyTorch + CUDA + Python
- Docker: not available
- System disk: `/`
- Data disk: `/root/rivermind-data`

Disk behavior:

| Path | Type | Note |
|---|---|---|
| `/` | system disk | limited capacity, not suitable for model cache |
| `/root/rivermind-data` | data disk | expandable, faster, suitable for project files, model cache, experiment outputs |

Because Docker was not available, the deployment uses pip/uv-based Python virtual environments instead of Docker containers.

---

## 2. Basic Server Setup

After connecting to the server via SSH:

```bash
ssh root@YOUR_SERVER_IP -p YOUR_PORT
```

Check GPU and environment:

```
nvidia-smi
python3 --version
df -h
```

Install common tools:

```
apt update
apt install -y git curl wget tmux htop nvtop rsync iproute2 unzip
```

Useful monitoring commands:

```
nvidia-smi
watch -n 2 nvidia-smi
nvtop
htop
ss -lntp | grep 8000 || true
ss -lntp | grep 30000 || true
```

---

## 3. GitHub SSH Setup

Generate SSH key:

```
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Show public key:

```
cat ~/.ssh/id_ed25519.pub
```

Add this key to GitHub:

```
GitHub → Settings → SSH and GPG keys → New SSH key
```

Test connection:

```
ssh -T git@github.com
```

Configure Git identity:

```
git config --global user.name "ChallenXiao"
git config --global user.email "your_email@example.com"
```

Clone project to data disk:

```
cd /root/rivermind-data
git clone git@github.com:ChallenXiao/llm-serving-kv-benchmark.git
cd llm-serving-kv-benchmark
```

If SSH is not ready, use HTTPS:

```
git clone https://github.com/ChallenXiao/llm-serving-kv-benchmark.git
```

---

## 4. Move Model Cache to Data Disk

The default Hugging Face cache may be under:

```
/root/.cache/huggingface
```

This can fill the system disk. Move cache to the data disk:

```
mkdir -p /root/rivermind-data/hf_cache
mkdir -p /root/rivermind-data/pip_cache
mkdir -p /root/rivermind-data/uv_cache
```

Copy existing cache if it already exists:

```
if [ -d /root/.cache/huggingface ]; then
  rsync -a /root/.cache/huggingface/ /root/rivermind-data/hf_cache/
fi
```

Set environment variables:

```
cat >> ~/.bashrc <<'EOF'

# LLM benchmark cache paths on data disk
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/rivermind-data/hf_cache
export HF_HUB_CACHE=/root/rivermind-data/hf_cache/hub
export TRANSFORMERS_CACHE=/root/rivermind-data/hf_cache/hub
export PIP_CACHE_DIR=/root/rivermind-data/pip_cache
export UV_CACHE_DIR=/root/rivermind-data/uv_cache
EOF

source ~/.bashrc
```

Create symlink so libraries that still use the default path are redirected:

```
mkdir -p /root/.cache

if [ -d /root/.cache/huggingface ] && [ ! -L /root/.cache/huggingface ]; then
  mv /root/.cache/huggingface /root/.cache/huggingface.bak_$(date +%Y%m%d_%H%M%S)
fi

ln -sfn /root/rivermind-data/hf_cache /root/.cache/huggingface
```

Check disk usage:

```
echo $HF_HOME
echo $HF_HUB_CACHE
du -sh /root/rivermind-data/hf_cache || true
df -h
```

After confirming the model cache has been moved, old backups can be removed carefully:

```
ls -lh /root/.cache
# Only remove after verification:
# rm -rf /root/.cache/huggingface.bak_YYYYMMDD_HHMMSS
```

---

## 5. Cache Cleanup Commands

Clean pip cache:

```
pip cache purge
```

Clean uv cache:

```
uv cache clean
```

Check large files:

```
du -h --max-depth=1 /root | sort -h
du -h --max-depth=1 /root/rivermind-data | sort -h
du -h --max-depth=1 /root/.cache | sort -h
```

Remove temporary experiment archives if needed:

```
rm -f experiments/*.tar.gz
```

---

## 6. vLLM Virtual Environment

A separate virtual environment is used for vLLM:

```
cd /root/rivermind-data/llm-serving-kv-benchmark

python3 -m venv .venv-gpu
source .venv-gpu/bin/activate

pip install --upgrade pip
pip install uv
pip install -r requirements-dev.txt
pip install matplotlib pandas numpy
```

Important compatibility note:

The first automatic vLLM installation selected a CUDA 13-dependent wheel and failed with:

```
ImportError: libcudart.so.13: cannot open shared object file
```

Fix: reinstall vLLM with CUDA 12.8 backend:

```
uv pip install "vllm==0.10.2" --torch-backend=cu128
```

The tokenizer then failed due to transformers compatibility:

```
AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
```

Fix:

```
uv pip uninstall -y transformers tokenizersuv pip install "transformers==4.56.1" "tokenizers>=0.22,<0.23"
```

Verify:

```
python - <<'PY'
import torch
import vllm
import transformers
import tokenizers

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
print("tokenizers:", tokenizers.__version__)
PY
```

Expected:

```
torch: 2.8.0+cu128
cuda available: True
gpu: NVIDIA GeForce RTX 3090
vllm: 0.10.2
```

---

## 7. Launch vLLM Server

Use data-disk cache and mirror:

```
source ~/.bashrc
source .venv-gpu/bin/activate
```

Prefix cache ON:

```
MODEL=Qwen/Qwen2.5-7B-Instruct \
PORT=8000 \
MAX_MODEL_LEN=4096 \
GPU_MEM_UTIL=0.88 \
MAX_NUM_SEQS=32 \
MAX_NUM_BATCHED_TOKENS=8192 \
PREFIX_CACHE=1 \
bash scripts/launch_vllm_pip.sh
```

Prefix cache OFF:

```
MODEL=Qwen/Qwen2.5-7B-Instruct \
PORT=8000 \
MAX_MODEL_LEN=4096 \
GPU_MEM_UTIL=0.88 \
MAX_NUM_SEQS=32 \
MAX_NUM_BATCHED_TOKENS=8192 \
PREFIX_CACHE=0 \
bash scripts/launch_vllm_pip.sh
```

Health check from another terminal:

```
curl http://127.0.0.1:8000/v1/models
```

Chat completion check:

```
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 32,
    "temperature": 0
  }'
```

Stop vLLM:

```
pkill -f "vllm" || true
ss -lntp | grep 8000 || true
```

---

## 8. Completed vLLM Experiments

### 8.1 Smoke Test

Command:

```
ENGINE=vllm_smoke \
BASE_URL=http://127.0.0.1:8000/v1 \
MODEL=Qwen/Qwen2.5-7B-Instruct \
SHORT_CONCURRENCIES="1 2" \
LONG_CONCURRENCIES="1" \
SHARED_CONCURRENCIES="1" \
NUM_REQUESTS_SHORT=4 \
NUM_REQUESTS_LONG=2 \
NUM_REQUESTS_SHARED=2 \
SHORT_INPUT_TOKENS=128 \
LONG_INPUT_TOKENS=512 \
SHARED_PREFIX_TOKENS=512 \
SHORT_MAX_TOKENS=32 \
LONG_MAX_TOKENS=32 \
SHARED_MAX_TOKENS=32 \
bash scripts/run_cloud_sweep.sh
```

Result:

- vLLM API server worked.
- `/v1/models` returned the model.
- `/v1/chat/completions` generated valid output.
- The benchmark client successfully collected TTFT, E2E latency, throughput, and raw JSONL results.
- A missing `matplotlib` dependency was fixed by installing plotting dependencies.

---

### 8.2 Preliminary vLLM Prefix Cache ON/OFF Sweep

Shared concurrency sweep:

```
1 / 2 / 4 / 8 / 16 / 32
```

Workloads:

```
short prompt
long prompt
shared prefix
```

Command template:

```
ENGINE=vllm_prefix_on \
BASE_URL=http://127.0.0.1:8000/v1 \
MODEL=Qwen/Qwen2.5-7B-Instruct \
SHORT_CONCURRENCIES="1 2 4 8 16 32" \
LONG_CONCURRENCIES="1 2 4 8 16 32" \
SHARED_CONCURRENCIES="1 2 4 8 16 32" \
NUM_REQUESTS_SHORT=96 \
NUM_REQUESTS_LONG=64 \
NUM_REQUESTS_SHARED=96 \
SHORT_INPUT_TOKENS=256 \
LONG_INPUT_TOKENS=1024 \
SHARED_PREFIX_TOKENS=1024 \
SHORT_MAX_TOKENS=128 \
LONG_MAX_TOKENS=128 \
SHARED_MAX_TOKENS=128 \
bash scripts/run_cloud_sweep.sh
```

Observed preliminary result:

At concurrency 32:

|Workload|Prefix Cache|Request Throughput|TTFT P95|
|---|---|---|---|
|long|ON|13.31 req/s|0.17s|
|long|OFF|3.21 req/s|7.22s|
|shared_prefix|ON|9.56 req/s|0.22s|
|shared_prefix|OFF|3.03 req/s|7.22s|
|short|ON|10.55 req/s|0.13s|
|short|OFF|6.49 req/s|1.97s|

Interpretation:

- Prefix caching significantly reduced TTFT under high concurrency.
- Prefix cache hit rate was observed to be very high.
- GPU KV cache usage was still low, which means the current workload did not fully stress KV memory capacity.
- This run is useful, but not fully clean because `NUM_REQUESTS_LONG=64` while short/shared used 96.

---

## 9. Why Prefix Cache Hit Rate Can Be High While KV Cache Usage Is Low

Example vLLM log:

```
GPU KV cache usage: 3.7%
Prefix cache hit rate: 99.3%
```

These two metrics measure different things:

|Metric|Meaning|
|---|---|
|Prefix cache hit rate|How much of the prompt prefix was reused from cached KV blocks|
|GPU KV cache usage|How much of the total allocated KV cache pool is currently occupied|

Therefore:

```
High prefix hit rate means repeated prefixes are being reused effectively.Low KV usage means the workload has not yet filled the available KV cache pool.
```

To increase KV cache pressure, use longer prompts, higher concurrency, larger output tokens, larger `max_num_seqs`, and larger `max_num_batched_tokens`.

---

## 10. Planned Clean Fair Comparison

The next fair comparison should use identical request counts for all workloads:

```
concurrency = 1 / 2 / 4 / 8 / 16 / 32
num_requests = 96 / 96 / 96
short input = 256
long input = 1024
shared prefix = 1024
output = 128
```

### vLLM Prefix ON Fair Run

Start server:

```
pkill -f "vllm" || true

source .venv-gpu/bin/activate
source ~/.bashrc

MODEL=Qwen/Qwen2.5-7B-Instruct \
PORT=8000 \
MAX_MODEL_LEN=4096 \
GPU_MEM_UTIL=0.88 \
MAX_NUM_SEQS=32 \
MAX_NUM_BATCHED_TOKENS=8192 \
PREFIX_CACHE=1 \
bash scripts/launch_vllm_pip.sh
```

Run benchmark:

```
ENGINE=vllm_prefix_on_fair \
BASE_URL=http://127.0.0.1:8000/v1 \
MODEL=Qwen/Qwen2.5-7B-Instruct \
SHORT_CONCURRENCIES="1 2 4 8 16 32" \
LONG_CONCURRENCIES="1 2 4 8 16 32" \
SHARED_CONCURRENCIES="1 2 4 8 16 32" \
NUM_REQUESTS_SHORT=96 \
NUM_REQUESTS_LONG=96 \
NUM_REQUESTS_SHARED=96 \
SHORT_INPUT_TOKENS=256 \
LONG_INPUT_TOKENS=1024 \
SHARED_PREFIX_TOKENS=1024 \
SHORT_MAX_TOKENS=128 \
LONG_MAX_TOKENS=128 \
SHARED_MAX_TOKENS=128 \
bash scripts/run_cloud_sweep.sh
```

### vLLM Prefix OFF Fair Run

Start server:

```
pkill -f "vllm" || true

source .venv-gpu/bin/activate
source ~/.bashrc

MODEL=Qwen/Qwen2.5-7B-Instruct \
PORT=8000 \
MAX_MODEL_LEN=4096 \
GPU_MEM_UTIL=0.88 \
MAX_NUM_SEQS=32 \
MAX_NUM_BATCHED_TOKENS=8192 \
PREFIX_CACHE=0 \
bash scripts/launch_vllm_pip.sh
```

Run benchmark:

```
ENGINE=vllm_prefix_off_fair \
BASE_URL=http://127.0.0.1:8000/v1 \
MODEL=Qwen/Qwen2.5-7B-Instruct \
SHORT_CONCURRENCIES="1 2 4 8 16 32" \
LONG_CONCURRENCIES="1 2 4 8 16 32" \
SHARED_CONCURRENCIES="1 2 4 8 16 32" \
NUM_REQUESTS_SHORT=96 \
NUM_REQUESTS_LONG=96 \
NUM_REQUESTS_SHARED=96 \
SHORT_INPUT_TOKENS=256 \
LONG_INPUT_TOKENS=1024 \
SHARED_PREFIX_TOKENS=1024 \
SHORT_MAX_TOKENS=128 \
LONG_MAX_TOKENS=128 \
SHARED_MAX_TOKENS=128 \
bash scripts/run_cloud_sweep.sh
```

---

## 11. Planned KV Pressure Experiment

Purpose:

```
Increase GPU KV cache usage and show how long prompt / high concurrency stresses KV cache memory.
```

Start vLLM with stronger settings:

```
pkill -f "vllm" || true

source .venv-gpu/bin/activate
source ~/.bashrc

MODEL=Qwen/Qwen2.5-7B-Instruct \
PORT=8000 \
MAX_MODEL_LEN=4096 \
GPU_MEM_UTIL=0.92 \
MAX_NUM_SEQS=64 \
MAX_NUM_BATCHED_TOKENS=16384 \
PREFIX_CACHE=1 \
bash scripts/launch_vllm_pip.sh
```

Run pressure benchmark:

```
ENGINE=vllm_kv_pressure \
BASE_URL=http://127.0.0.1:8000/v1 \
MODEL=Qwen/Qwen2.5-7B-Instruct \
SHORT_CONCURRENCIES="1 2 4 8 16 32 64" \
LONG_CONCURRENCIES="1 2 4 8 16 32 64" \
SHARED_CONCURRENCIES="1 2 4 8 16 32 64" \
NUM_REQUESTS_SHORT=128 \
NUM_REQUESTS_LONG=128 \
NUM_REQUESTS_SHARED=128 \
SHORT_INPUT_TOKENS=512 \
LONG_INPUT_TOKENS=3072 \
SHARED_PREFIX_TOKENS=3072 \
SHORT_MAX_TOKENS=256 \
LONG_MAX_TOKENS=256 \
SHARED_MAX_TOKENS=256 \
bash scripts/run_cloud_sweep.sh
```

Watch server logs:

```
watch -n 2 nvidia-smi
```

Look for vLLM logs such as:

```
GPU KV cache usage
Prefix cache hit rate
Running reqs
Waiting reqs
Avg prompt throughput
Avg generation throughput
```

If the server OOMs, reduce:

```
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=8192
LONG_INPUT_TOKENS=2048
SHARED_PREFIX_TOKENS=2048
```

---

## 12. Planned SGLang Environment

A separate virtual environment should be used for SGLang to avoid breaking the working vLLM environment:

```
cd /root/rivermind-data/llm-serving-kv-benchmark

python3 -m venv .venv-sglang
source .venv-sglang/bin/activate
source ~/.bashrc

pip install --upgrade pip
pip install uv
pip install -r requirements-dev.txt
pip install matplotlib pandas numpy
```

Install PyTorch CUDA 12.8:

```
uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

Install SGLang:

```
uv pip install "sglang[all]"
```

If this fails:

```
uv pip install sglang
```

Verify:

```
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

import sglang
print("sglang import ok")
PY
```

---

## 13. Planned SGLang Fair Experiments

### SGLang RadixAttention ON

Start server:

```
pkill -f "vllm" || true
pkill -f "sglang" || true

cd /root/rivermind-data/llm-serving-kv-benchmark
source .venv-sglang/bin/activate
source ~/.bashrc

MODEL=Qwen/Qwen2.5-7B-Instruct \
PORT=30000 \
MAX_MODEL_LEN=4096 \
MEM_FRACTION=0.82 \
MAX_RUNNING_REQUESTS=32 \
MAX_PREFILL_TOKENS=8192 \
RADIX_CACHE=1 \
bash scripts/launch_sglang_pip.sh
```

Run benchmark:

```
ENGINE=sglang_radix_on_fair \
BASE_URL=http://127.0.0.1:30000/v1 \
MODEL=Qwen/Qwen2.5-7B-Instruct \
SHORT_CONCURRENCIES="1 2 4 8 16 32" \
LONG_CONCURRENCIES="1 2 4 8 16 32" \
SHARED_CONCURRENCIES="1 2 4 8 16 32" \
NUM_REQUESTS_SHORT=96 \
NUM_REQUESTS_LONG=96 \
NUM_REQUESTS_SHARED=96 \
SHORT_INPUT_TOKENS=256 \
LONG_INPUT_TOKENS=1024 \
SHARED_PREFIX_TOKENS=1024 \
SHORT_MAX_TOKENS=128 \
LONG_MAX_TOKENS=128 \
SHARED_MAX_TOKENS=128 \
bash scripts/run_cloud_sweep.sh
```

### SGLang RadixAttention OFF

Start server:

```
pkill -f "sglang" || true

cd /root/rivermind-data/llm-serving-kv-benchmark
source .venv-sglang/bin/activate
source ~/.bashrc

MODEL=Qwen/Qwen2.5-7B-Instruct \
PORT=30000 \
MAX_MODEL_LEN=4096 \
MEM_FRACTION=0.82 \
MAX_RUNNING_REQUESTS=32 \
MAX_PREFILL_TOKENS=8192 \
RADIX_CACHE=0 \
bash scripts/launch_sglang_pip.sh
```

Run benchmark:

```
ENGINE=sglang_radix_off_fair \
BASE_URL=http://127.0.0.1:30000/v1 \
MODEL=Qwen/Qwen2.5-7B-Instruct \
SHORT_CONCURRENCIES="1 2 4 8 16 32" \
LONG_CONCURRENCIES="1 2 4 8 16 32" \
SHARED_CONCURRENCIES="1 2 4 8 16 32" \
NUM_REQUESTS_SHORT=96 \
NUM_REQUESTS_LONG=96 \
NUM_REQUESTS_SHARED=96 \
SHORT_INPUT_TOKENS=256 \
LONG_INPUT_TOKENS=1024 \
SHARED_PREFIX_TOKENS=1024 \
SHORT_MAX_TOKENS=128 \
LONG_MAX_TOKENS=128 \
SHARED_MAX_TOKENS=128 \
bash scripts/run_cloud_sweep.sh
```

---

## 14. Experiment Output Structure

Each run creates:

```
experiments/<engine>_<timestamp>/
├── raw/
│   └── *.jsonl
├── processed/
│   └── summary.csv
├── plots/
│   ├── request_throughput_vs_concurrency.png
│   ├── output_throughput_vs_concurrency.png
│   ├── p95_latency_vs_concurrency.png
│   └── p95_ttft_vs_concurrency.png
├── logs/
└── run_config.txt
```

The most important files are:

```
experiments/*/processed/summary.csv
experiments/*/plots/*.png
experiments/*/raw/*.jsonl
```

---

## 15. Mapping to Resume Claims

|Project Evidence|Resume Claim|
|---|---|
|vLLM server on RTX 3090|Deployed OpenAI-compatible LLM serving engine on cloud GPU|
|`run_cloud_sweep.sh`|Built custom async benchmark pipeline|
|short / long / shared-prefix workloads|Designed workload-specific serving experiments|
|TTFT / TPOT / P95 / throughput metrics|Analyzed prefill/decode and latency-throughput tradeoff|
|vLLM prefix on/off|Quantified prefix cache effect on TTFT and throughput|
|KV pressure experiment|Studied KV cache pressure under long context and high concurrency|
|`kv_cache_sim/`|Implemented PagedAttention-style KV cache allocator simulator|
|planned SGLang radix on/off|Planned RadixAttention comparison under the same workload|
