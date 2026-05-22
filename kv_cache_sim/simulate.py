from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kv_cache_sim.contiguous_allocator import ContiguousAllocator
from kv_cache_sim.paged_allocator import PagedAllocator
from kv_cache_sim.prefix_reuse import simulate_prefix_reuse


def generate_request_lengths(
    num_requests: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)

    # Long-tail distribution: many short requests, some long requests.
    raw = rng.lognormal(mean=6.0, sigma=0.9, size=num_requests)
    raw = raw / raw.max()

    lengths = min_len + raw * (max_len - min_len)
    return [int(x) for x in lengths]


def run_allocator_simulation(args: argparse.Namespace) -> dict:
    lengths = generate_request_lengths(
        num_requests=args.num_requests,
        min_len=args.min_len,
        max_len=args.max_seq_len,
        seed=args.seed,
    )

    contiguous = ContiguousAllocator(max_seq_len=args.max_seq_len)
    paged = PagedAllocator(block_size=args.block_size)

    for request_id, actual_tokens in enumerate(lengths):
        contiguous.allocate(request_id=request_id, actual_tokens=actual_tokens)
        paged.allocate(request_id=request_id, actual_tokens=actual_tokens)

    prefix_result = simulate_prefix_reuse(
        num_requests=args.num_requests,
        shared_prefix_tokens=args.shared_prefix_tokens,
        unique_suffix_tokens=args.unique_suffix_tokens,
        block_size=args.block_size,
    )

    return {
        "config": {
            "num_requests": args.num_requests,
            "min_len": args.min_len,
            "max_seq_len": args.max_seq_len,
            "block_size": args.block_size,
            "seed": args.seed,
        },
        "request_lengths": lengths,
        "contiguous": contiguous.summary(),
        "paged": paged.summary(),
        "prefix_reuse": prefix_result,
    }


def save_outputs(result: dict, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "kv_cache_sim_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    rows = [
        result["contiguous"],
        result["paged"],
    ]

    df = pd.DataFrame(rows)
    df.to_csv(output_path / "kv_cache_allocator_summary.csv", index=False)

    lengths_df = pd.DataFrame(
        {
            "request_id": list(range(len(result["request_lengths"]))),
            "actual_tokens": result["request_lengths"],
        }
    )
    lengths_df.to_csv(output_path / "request_lengths.csv", index=False)


def print_result(result: dict) -> None:
    print("\n========== KV Cache Allocator Simulation ==========")
    print(json.dumps(result["config"], indent=2))

    print("\n--- Contiguous Allocation ---")
    for k, v in result["contiguous"].items():
        print(f"{k}: {v}")

    print("\n--- Paged Allocation ---")
    for k, v in result["paged"].items():
        print(f"{k}: {v}")

    print("\n--- Prefix Reuse ---")
    for k, v in result["prefix_reuse"].items():
        print(f"{k}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--min-len", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--shared-prefix-tokens", type=int, default=1024)
    parser.add_argument("--unique-suffix-tokens", type=int, default=128)

    parser.add_argument("--output-dir", type=str, default="results/processed")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_allocator_simulation(args)
    print_result(result)
    save_outputs(result, args.output_dir)

    print(f"\nSaved simulation outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
