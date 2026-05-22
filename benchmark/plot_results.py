from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from benchmark.metrics import summarize_results


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def infer_metadata_from_filename(path: Path) -> dict[str, Any]:
    """
    Infer metadata from filenames like:
    mock_short_c4.jsonl
    vllm_long_c8.jsonl
    sglang_shared_prefix_c16.jsonl
    """
    name = path.stem

    match = re.match(r"(?P<engine>.+?)_(?P<workload>short|long|shared_prefix)_c(?P<concurrency>\d+)", name)

    if not match:
        return {
            "engine": "unknown",
            "workload": "unknown",
            "concurrency": -1,
        }

    return {
        "engine": match.group("engine"),
        "workload": match.group("workload"),
        "concurrency": int(match.group("concurrency")),
    }


def summarize_file(path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    summary = summarize_results(records)

    metadata = infer_metadata_from_filename(path)

    if records:
        first = records[0]
        metadata["engine"] = first.get("engine", metadata["engine"])
        metadata["workload"] = first.get("workload", metadata["workload"])
        metadata["concurrency"] = int(first.get("concurrency", metadata["concurrency"]))
        metadata["model"] = first.get("model", "unknown")
        metadata["max_tokens"] = first.get("max_tokens", None)
        metadata["input_tokens"] = first.get("input_tokens", None)
    else:
        metadata["model"] = "unknown"
        metadata["max_tokens"] = None
        metadata["input_tokens"] = None

    return {
        "file": str(path),
        **metadata,
        **summary,
    }


def load_all_summaries(raw_dir: Path) -> pd.DataFrame:
    rows = []

    for path in sorted(raw_dir.glob("*.jsonl")):
        rows.append(summarize_file(path))

    if not rows:
        raise FileNotFoundError(f"No jsonl files found in {raw_dir}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["engine", "workload", "concurrency"])
    return df


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 6))

    grouped = df.groupby(["engine", "workload"], dropna=False)

    for (engine, workload), group in grouped:
        group = group.sort_values("concurrency")
        label = f"{engine}-{workload}"
        plt.plot(group["concurrency"], group[metric], marker="o", label=label)

    plt.xlabel("Concurrency")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs Concurrency")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="results/raw")
    parser.add_argument("--summary-output", type=str, default="results/processed/summary.csv")
    parser.add_argument("--plot-dir", type=str, default="plots")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    summary_output = Path(args.summary_output)
    plot_dir = Path(args.plot_dir)

    df = load_all_summaries(raw_dir)

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_output, index=False)

    print("\n========== Summary Table ==========")
    print(
        df[
            [
                "engine",
                "workload",
                "concurrency",
                "num_successful",
                "request_throughput_req_s",
                "output_throughput_tok_s",
                "e2e_latency_p95_s",
                "ttft_p95_s",
                "tpot_mean_s",
            ]
        ].to_string(index=False)
    )

    plot_metric(
        df=df,
        metric="request_throughput_req_s",
        ylabel="Request Throughput (req/s)",
        output_path=plot_dir / "request_throughput_vs_concurrency.png",
    )

    plot_metric(
        df=df,
        metric="output_throughput_tok_s",
        ylabel="Output Throughput (tokens/s)",
        output_path=plot_dir / "output_throughput_vs_concurrency.png",
    )

    plot_metric(
        df=df,
        metric="e2e_latency_p95_s",
        ylabel="P95 End-to-End Latency (s)",
        output_path=plot_dir / "p95_latency_vs_concurrency.png",
    )

    plot_metric(
        df=df,
        metric="ttft_p95_s",
        ylabel="P95 TTFT (s)",
        output_path=plot_dir / "p95_ttft_vs_concurrency.png",
    )

    print(f"\nSaved summary to: {summary_output}")
    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()
