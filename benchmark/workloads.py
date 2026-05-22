from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


WorkloadType = Literal["short", "long", "shared_prefix"]


@dataclass
class WorkloadItem:
    request_id: int
    messages: list[dict[str, str]]
    max_tokens: int


def make_repeated_text(topic: str, target_words: int) -> str:
    """
    Roughly generate text with a controllable length.
    This is not exact tokenization, but enough for local benchmark logic.
    """
    words = []
    for i in range(target_words):
        words.append(f"{topic}_{i}")
    return " ".join(words)


def build_short_prompt(request_id: int, input_tokens: int, max_tokens: int) -> WorkloadItem:
    user_text = (
        "Summarize the following short business note and give one recommendation. "
        + make_repeated_text("short_context", max(8, input_tokens - 20))
    )

    return WorkloadItem(
        request_id=request_id,
        messages=[
            {"role": "system", "content": "You are a concise and helpful assistant."},
            {"role": "user", "content": user_text},
        ],
        max_tokens=max_tokens,
    )


def build_long_prompt(request_id: int, input_tokens: int, max_tokens: int) -> WorkloadItem:
    user_text = (
        "Read the following long document and extract key risks, opportunities, and conclusions. "
        + make_repeated_text("long_context", max(32, input_tokens - 20))
    )

    return WorkloadItem(
        request_id=request_id,
        messages=[
            {"role": "system", "content": "You are a rigorous document analysis assistant."},
            {"role": "user", "content": user_text},
        ],
        max_tokens=max_tokens,
    )


def build_shared_prefix_prompt(
    request_id: int,
    shared_prefix_tokens: int,
    max_tokens: int,
) -> WorkloadItem:
    shared_system_prompt = (
        "You are a rigorous financial analysis assistant. "
        "You must analyze revenue growth, margin quality, capital structure, risk factors, "
        "and valuation implications carefully. "
        + make_repeated_text("shared_financial_policy", max(64, shared_prefix_tokens))
    )

    user_questions = [
        "Analyze the revenue growth trend and identify the key drivers.",
        "Evaluate the company's profitability and margin sustainability.",
        "Assess the balance sheet risk and debt structure.",
        "Give a short investment-style conclusion.",
        "Compare the business model with its closest competitor.",
    ]

    user_text = user_questions[request_id % len(user_questions)]

    return WorkloadItem(
        request_id=request_id,
        messages=[
            {"role": "system", "content": shared_system_prompt},
            {"role": "user", "content": user_text},
        ],
        max_tokens=max_tokens,
    )


def build_workload(
    workload: WorkloadType,
    num_requests: int,
    input_tokens: int,
    max_tokens: int,
    shared_prefix_tokens: int = 1024,
) -> list[WorkloadItem]:
    items: list[WorkloadItem] = []

    for i in range(num_requests):
        if workload == "short":
            items.append(build_short_prompt(i, input_tokens=input_tokens, max_tokens=max_tokens))
        elif workload == "long":
            items.append(build_long_prompt(i, input_tokens=input_tokens, max_tokens=max_tokens))
        elif workload == "shared_prefix":
            items.append(
                build_shared_prefix_prompt(
                    i,
                    shared_prefix_tokens=shared_prefix_tokens,
                    max_tokens=max_tokens,
                )
            )
        else:
            raise ValueError(f"Unknown workload: {workload}")

    return items
