from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RequestAllocation:
    request_id: int
    actual_tokens: int
    reserved_tokens: int


class ContiguousAllocator:
    """
    Simulate naive contiguous KV cache allocation.

    Each request reserves max_seq_len tokens regardless of its actual length.
    This is simple but can waste a lot of memory when sequence lengths vary.
    """

    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.allocations: list[RequestAllocation] = []

    def allocate(self, request_id: int, actual_tokens: int) -> None:
        if actual_tokens > self.max_seq_len:
            raise ValueError(
                f"actual_tokens={actual_tokens} exceeds max_seq_len={self.max_seq_len}"
            )

        self.allocations.append(
            RequestAllocation(
                request_id=request_id,
                actual_tokens=actual_tokens,
                reserved_tokens=self.max_seq_len,
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
    def utilization(self) -> float:
        if self.reserved_tokens == 0:
            return 0.0
        return self.used_tokens / self.reserved_tokens

    def summary(self) -> dict:
        return {
            "strategy": "contiguous",
            "num_requests": len(self.allocations),
            "used_tokens": self.used_tokens,
            "reserved_tokens": self.reserved_tokens,
            "wasted_tokens": self.wasted_tokens,
            "utilization": self.utilization,
        }
