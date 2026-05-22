from __future__ import annotations

from typing import Any

import numpy as np


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [r for r in records if r.get("error") is None]

    if not records:
        return {"error": "no records"}

    total_duration_s = max(r["finish_time_s"] for r in records) - min(
        r["start_time_s"] for r in records
    )

    if total_duration_s <= 0:
        total_duration_s = 1e-9

    e2e = [r["e2e_latency_s"] for r in successful]
    ttft = [r["ttft_s"] for r in successful if r["ttft_s"] is not None]
    output_tokens = [r["approx_output_tokens"] for r in successful]

    tpot = []
    for r in successful:
        if r["ttft_s"] is not None and r["approx_output_tokens"] > 1:
            decode_time = r["e2e_latency_s"] - r["ttft_s"]
            tpot.append(decode_time / (r["approx_output_tokens"] - 1))

    total_output_tokens = sum(output_tokens)

    return {
        "num_requests": len(records),
        "num_successful": len(successful),
        "num_failed": len(records) - len(successful),
        "total_duration_s": total_duration_s,
        "request_throughput_req_s": len(successful) / total_duration_s,
        "output_throughput_tok_s": total_output_tokens / total_duration_s,
        "e2e_latency_p50_s": percentile(e2e, 50),
        "e2e_latency_p95_s": percentile(e2e, 95),
        "e2e_latency_p99_s": percentile(e2e, 99),
        "ttft_p50_s": percentile(ttft, 50),
        "ttft_p95_s": percentile(ttft, 95),
        "ttft_p99_s": percentile(ttft, 99),
        "tpot_mean_s": mean(tpot),
        "approx_total_output_tokens": total_output_tokens,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n========== Benchmark Summary ==========")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
