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
