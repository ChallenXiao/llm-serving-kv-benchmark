from __future__ import annotations

import json
import time
from typing import Any

import httpx

from benchmark.workloads import WorkloadItem


def count_approx_tokens(text: str) -> int:
    """
    Approximate token count for local testing.
    Later, we can replace this with a real tokenizer.
    """
    if not text:
        return 0
    return len(text.split())


async def run_one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    item: WorkloadItem,
    temperature: float = 0.0,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"

    payload = {
        "model": model,
        "messages": item.messages,
        "max_tokens": item.max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    start_time = time.perf_counter()
    first_token_time: float | None = None
    output_text_parts: list[str] = []
    error: str | None = None

    try:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code >= 400:
                body = await response.aread()
                error_body = body.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"HTTP {response.status_code}: {error_body[:1500]}"
                )

            async for line in response.aiter_lines():
                if not line:
                    continue

                if not line.startswith("data:"):
                    continue

                data = line.removeprefix("data:").strip()

                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")

                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    output_text_parts.append(content)

    except Exception as exc:
        error = repr(exc)

    finish_time = time.perf_counter()
    output_text = "".join(output_text_parts)

    return {
        "request_id": item.request_id,
        "start_time_s": start_time,
        "first_token_time_s": first_token_time,
        "finish_time_s": finish_time,
        "ttft_s": None if first_token_time is None else first_token_time - start_time,
        "e2e_latency_s": finish_time - start_time,
        "approx_output_tokens": count_approx_tokens(output_text),
        "output_chars": len(output_text),
        "error": error,
    }