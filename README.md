# LLM Serving Benchmark and KV Cache Scheduling Analysis with vLLM and SGLang

This project evaluates LLM serving performance across vLLM and SGLang under different workloads, with a focus on throughput, latency, prefill/decode behavior, prefix caching, and KV cache memory management.

## Tech Stack

- Python
- vLLM
- SGLang
- OpenAI-compatible API
- Async benchmarking
- KV cache allocator simulation
- Docker
- Qwen2.5-7B-Instruct

## Goals

1. Deploy OpenAI-compatible LLM serving engines with vLLM and SGLang.
2. Build custom workloads for short prompts, long prompts, and shared-prefix scenarios.
3. Measure request throughput, output tokens/s, TTFT, TPOT, ITL, and P95/P99 latency.
4. Compare prefix caching behavior in vLLM and SGLang.
5. Implement a KV cache block allocator simulator inspired by PagedAttention.

## Workloads

| Workload | Description | Main Bottleneck |
|---|---|---|
| A | Short prompt, high concurrency | Decode / scheduling |
| B | Long prompt, fixed output | Prefill |
| C | Shared long system prompt | Prefix cache reuse |
| D | Speculative decoding | Optional |

## Current Status

- [ ] Repo scaffold
- [ ] Local async benchmark client
- [ ] Mock OpenAI-compatible server
- [ ] vLLM launch script
- [ ] SGLang launch script
- [ ] Official benchmark results
- [ ] Custom benchmark results
- [ ] KV cache simulator
- [ ] Final plots and analysis

## Local Benchmark Validation

Before running expensive GPU experiments, I implemented a local OpenAI-compatible mock server to validate the full benchmarking pipeline.

The local pipeline supports:

- streaming response parsing
- first-token timestamp collection
- end-to-end latency measurement
- approximate output token counting
- concurrency control with asyncio
- JSONL result export
- automatic summary CSV generation
- throughput and latency plotting

This allows the benchmark client to be debugged locally on a MacBook before being reused against real vLLM and SGLang servers on a rented NVIDIA GPU instance.

## KV Cache Allocator Simulation

To better understand why KV cache scheduling matters for LLM serving, I implemented a small simulator comparing two memory allocation strategies:

1. **Contiguous pre-allocation**: each request reserves memory according to the maximum sequence length.
2. **Paged allocation**: each request dynamically allocates fixed-size blocks according to its actual sequence length.

In a simulated long-tail workload with 1,000 requests and a maximum sequence length of 4,096 tokens, contiguous allocation reserved 4,096,000 token slots but only used 396,344 of them, resulting in about 9.68% utilization. In contrast, paged allocation reserved 403,776 token slots with about 98.16% utilization.

I also simulated shared-prefix reuse. With 1,000 requests sharing a 1,024-token prefix and each having a 128-token unique suffix, prefix reuse reduced reserved token slots from 1,152,000 to 129,024, saving about 88.8% of KV cache memory.

## Key Figures

### Local Benchmark Validation

The following figures are generated from the local OpenAI-compatible mock server to validate the benchmarking pipeline before running GPU experiments.

![Request Throughput vs Concurrency](docs/images/request_throughput_vs_concurrency.png)

![P95 Latency vs Concurrency](docs/images/p95_latency_vs_concurrency.png)

![P95 TTFT vs Concurrency](docs/images/p95_ttft_vs_concurrency.png)

### KV Cache Allocator Simulation

The following figures show why KV cache memory management is important for LLM serving.

![KV Cache Allocator Utilization](docs/images/kv_allocator_utilization.png)

![KV Cache Reserved Tokens](docs/images/kv_allocator_reserved_tokens.png)

![Prefix Reuse Saving Ratio](docs/images/prefix_reuse_saving_ratio.png)

## Cloud GPU Deployment

Real vLLM and SGLang experiments are intended to be run on a rented NVIDIA GPU instance such as RTX 4090 / RTX 3090 / A10.

See:

- [AutoDL GPU Deployment Checklist](docs/autodl_gpu_deployment_checklist.md)

## One-Command Cloud Benchmark Sweep

After launching a vLLM or SGLang server, the full benchmark sweep can be executed with one command.

For vLLM:

```bash
ENGINE=vllm \
BASE_URL=http://127.0.0.1:8000/v1 \
MODEL=Qwen/Qwen2.5-7B-Instruct \
bash scripts/run_cloud_sweep.sh
```

For SGLang:

```bash
ENGINE=sglang \
BASE_URL=http://127.0.0.1:30000/v1 \
MODEL=Qwen/Qwen2.5-7B-Instruct \
bash scripts/run_cloud_sweep.sh
```

The script automatically runs short-prompt, long-prompt, and shared-prefix workloads across multiple concurrency levels, then saves raw JSONL results, summary CSV files, plots, logs, and a compressed experiment archive under experiments/.

See also:

notebooks/cloud_experiment_plan.ipynb
docs/autodl_gpu_deployment_checklist.md
