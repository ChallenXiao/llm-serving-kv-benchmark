from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class RequestAllocation:
    request_id: int
    actual_tokens: int
    num_blocks: int
    reserved_tokens: int


class PagedAllocator:
    """
    Simulate paged KV cache allocation.

    Each request is split into fixed-size blocks.
    Memory is allocated according to actual sequence length instead of max_seq_len.
    """

    def __init__(self, block_size: int):
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.block_size = block_size
        self.allocations: list[RequestAllocation] = []

    def allocate(self, request_id: int, actual_tokens: int) -> None:
        if actual_tokens <= 0:
            raise ValueError("actual_tokens must be positive")

        num_blocks = math.ceil(actual_tokens / self.block_size)
        reserved_tokens = num_blocks * self.block_size

        self.allocations.append(
            RequestAllocation(
                request_id=request_id,
                actual_tokens=actual_tokens,
                num_blocks=num_blocks,
                reserved_tokens=reserved_tokens,
            )
        )

    @property
    def used_tokens(self) -> int:
        return sum(a.actual_tokens for a in self.allocations)

    @property
    def reserved_tokens(self) -> int:
        return sum(a.reserved_tokens for a in self.allocations)

    @property
    def wasted_tokens(self) -> int:
        return self.reserved_tokens - self.used_tokens

    @property
    def total_blocks(self) -> int:
        return sum(a.num_blocks for a in self.allocations)

    @property
    def utilization(self) -> float:
        if self.reserved_tokens == 0:
            return 0.0
        return self.used_tokens / self.reserved_tokens

    def summary(self) -> dict:
        return {
            "strategy": "paged",
            "num_requests": len(self.allocations),
            "block_size": self.block_size,
            "total_blocks": self.total_blocks,
            "used_tokens": self.used_tokens,
            "reserved_tokens": self.reserved_tokens,
            "wasted_tokens": self.wasted_tokens,
            "utilization": self.utilization,
        }
