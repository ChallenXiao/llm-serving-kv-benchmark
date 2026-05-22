from __future__ import annotations

import math


def simulate_prefix_reuse(
    num_requests: int,
    shared_prefix_tokens: int,
    unique_suffix_tokens: int,
    block_size: int,
) -> dict:
    """
    Simulate memory usage with and without prefix reuse.

    Without prefix reuse:
        every request stores shared_prefix + unique_suffix.

    With prefix reuse:
        shared_prefix is stored once, and each request stores only its suffix.
    """

    if num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if shared_prefix_tokens < 0:
        raise ValueError("shared_prefix_tokens must be non-negative")
    if unique_suffix_tokens <= 0:
        raise ValueError("unique_suffix_tokens must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    total_tokens_per_request = shared_prefix_tokens + unique_suffix_tokens

    no_reuse_blocks_per_request = math.ceil(total_tokens_per_request / block_size)
    no_reuse_reserved_tokens = no_reuse_blocks_per_request * block_size * num_requests

    shared_prefix_blocks = math.ceil(shared_prefix_tokens / block_size)
    suffix_blocks_per_request = math.ceil(unique_suffix_tokens / block_size)

    with_reuse_reserved_tokens = (
        shared_prefix_blocks * block_size
        + suffix_blocks_per_request * block_size * num_requests
    )

    saved_tokens = no_reuse_reserved_tokens - with_reuse_reserved_tokens
    saving_ratio = saved_tokens / no_reuse_reserved_tokens if no_reuse_reserved_tokens > 0 else 0.0

    return {
        "num_requests": num_requests,
        "shared_prefix_tokens": shared_prefix_tokens,
        "unique_suffix_tokens": unique_suffix_tokens,
        "block_size": block_size,
        "no_reuse_reserved_tokens": no_reuse_reserved_tokens,
        "with_reuse_reserved_tokens": with_reuse_reserved_tokens,
        "saved_tokens": saved_tokens,
        "saving_ratio": saving_ratio,
    }
