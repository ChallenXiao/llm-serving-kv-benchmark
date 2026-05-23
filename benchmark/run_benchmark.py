from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from benchmark.async_client import run_one_request
from benchmark.metrics import print_summary, summarize_results
from benchmark.workloads import build_workload


async def run_benchmark(args: argparse.Namespace) -> list[dict]:
    workload_items = build_workload(
        workload=args.workload,
        num_requests=args.num_requests,
        input_tokens=args.input_tokens,
        max_tokens=args.max_tokens,
        shared_prefix_tokens=args.shared_prefix_tokens,
    )

    timeout = httpx.Timeout(args.timeout_s)
    limits = httpx.Limits(
        max_connections=max(args.concurrency * 2, 10),
        max_keepalive_connections=max(args.concurrency * 2, 10),
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    records: list[dict] = []

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

        async def bounded_run(item):
            async with semaphore:
                return await run_one_request(
                    client=client,
                    base_url=args.base_url,
                    model=args.model,
                    item=item,
                    temperature=args.temperature,
                )

        tasks = [asyncio.create_task(bounded_run(item)) for item in workload_items]

        completed = 0
        total = len(tasks)
        milestones = {
            max(1, int(total * 0.25)): "25%",
            max(1, int(total * 0.50)): "50%",
            max(1, int(total * 0.75)): "75%",
            total: "100%",
        }

        for task in asyncio.as_completed(tasks):
            record = await task

            # Add experiment metadata to every record.
            record.update(
                {
                    "engine": args.engine,
                    "model": args.model,
                    "workload": args.workload,
                    "concurrency": args.concurrency,
                    "num_requests": args.num_requests,
                    "input_tokens": args.input_tokens,
                    "shared_prefix_tokens": args.shared_prefix_tokens,
                    "max_tokens": args.max_tokens,
                }
            )

            records.append(record)
            completed += 1

            if completed in milestones:
                successful_so_far = sum(1 for r in records if r.get("error") is None)
                failed_so_far = completed - successful_so_far
                print(
                    f"progress {milestones[completed]}: "
                    f"{completed}/{total} completed, "
                    f"success={successful_so_far}, failed={failed_so_far}",
                    flush=True,
                )
            else:
                print(f"completed {completed}/{total} requests", end="\r")

    print()
    return records


def save_jsonl(records: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--engine", type=str, default="unknown")
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--workload", type=str, choices=["short", "long", "shared_prefix"], required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--shared-prefix-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = asyncio.run(run_benchmark(args))
    save_jsonl(records, args.output)

    summary = summarize_results(records)
    print_summary(summary)

    print(f"\nSaved raw results to: {args.output}")


if __name__ == "__main__":
    main()
