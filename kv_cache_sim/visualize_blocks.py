from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from kv_cache_sim.prefix_reuse import simulate_prefix_reuse


def load_allocator_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run python -m kv_cache_sim.simulate first."
        )
    return pd.read_csv(path)


def plot_allocator_utilization(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))

    labels = df["strategy"].tolist()
    values = (df["utilization"] * 100).tolist()

    plt.bar(labels, values)
    plt.ylabel("Memory Utilization (%)")
    plt.title("KV Cache Memory Utilization: Contiguous vs Paged")
    plt.ylim(0, 110)

    for i, value in enumerate(values):
        plt.text(i, value + 2, f"{value:.2f}%", ha="center")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_reserved_tokens(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))

    labels = df["strategy"].tolist()
    reserved = df["reserved_tokens"].tolist()
    wasted = df["wasted_tokens"].tolist()

    plt.bar(labels, reserved, label="Reserved tokens")
    plt.bar(labels, wasted, label="Wasted tokens")

    plt.ylabel("Token Slots")
    plt.title("KV Cache Reserved vs Wasted Token Slots")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_prefix_reuse_saving(
    block_size: int,
    shared_prefix_tokens: int,
    unique_suffix_tokens: int,
    output_path: Path,
) -> None:
    num_requests_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]

    rows = []
    for num_requests in num_requests_list:
        result = simulate_prefix_reuse(
            num_requests=num_requests,
            shared_prefix_tokens=shared_prefix_tokens,
            unique_suffix_tokens=unique_suffix_tokens,
            block_size=block_size,
        )
        rows.append(result)

    df = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    plt.plot(
        df["num_requests"],
        df["saving_ratio"] * 100,
        marker="o",
    )

    plt.xscale("log", base=2)
    plt.xlabel("Number of Requests")
    plt.ylabel("Memory Saving Ratio (%)")
    plt.title("Prefix Reuse Memory Saving vs Number of Requests")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    return df


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--allocator-summary",
        type=str,
        default="results/processed/kv_cache_allocator_summary.csv",
    )
    parser.add_argument("--plot-dir", type=str, default="plots")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--shared-prefix-tokens", type=int, default=1024)
    parser.add_argument("--unique-suffix-tokens", type=int, default=128)

    args = parser.parse_args()

    plot_dir = Path(args.plot_dir)
    allocator_summary_path = Path(args.allocator_summary)

    df = load_allocator_summary(allocator_summary_path)

    plot_allocator_utilization(
        df=df,
        output_path=plot_dir / "kv_allocator_utilization.png",
    )

    plot_reserved_tokens(
        df=df,
        output_path=plot_dir / "kv_allocator_reserved_tokens.png",
    )

    prefix_df = plot_prefix_reuse_saving(
        block_size=args.block_size,
        shared_prefix_tokens=args.shared_prefix_tokens,
        unique_suffix_tokens=args.unique_suffix_tokens,
        output_path=plot_dir / "prefix_reuse_saving_ratio.png",
    )

    prefix_df.to_csv(
        Path("results/processed/prefix_reuse_saving_ratio.csv"),
        index=False,
    )

    print("\nSaved KV cache simulation plots:")
    print(f"- {plot_dir / 'kv_allocator_utilization.png'}")
    print(f"- {plot_dir / 'kv_allocator_reserved_tokens.png'}")
    print(f"- {plot_dir / 'prefix_reuse_saving_ratio.png'}")
    print("\nSaved prefix reuse sweep:")
    print("- results/processed/prefix_reuse_saving_ratio.csv")


if __name__ == "__main__":
    main()
