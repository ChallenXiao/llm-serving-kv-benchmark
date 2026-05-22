from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel


app = FastAPI(title="Mock OpenAI-Compatible Server")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = False


def estimate_prompt_words(messages: list[dict[str, str]]) -> int:
    total = 0
    for message in messages:
        total += len(message.get("content", "").split())
    return total


def make_chunk(model: str, content: str, finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }


async def stream_response(req: ChatCompletionRequest):
    prompt_words = estimate_prompt_words(req.messages)

    # Simulate prefill cost: longer input -> slower first token.
    prefill_delay_s = min(0.8, 0.02 + prompt_words * 0.0004)
    await asyncio.sleep(prefill_delay_s)

    for i in range(req.max_tokens):
        # Simulate decode speed.
        await asyncio.sleep(0.01)
        token_text = f" tok{i}"
        chunk = make_chunk(req.model, token_text)
        yield f"data: {json.dumps(chunk)}\n\n"

    final_chunk = make_chunk(req.model, "", finish_reason="stop")
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        return StreamingResponse(stream_response(req), media_type="text/event-stream")

    content = " ".join([f"tok{i}" for i in range(req.max_tokens)])

    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": estimate_prompt_words(req.messages),
                "completion_tokens": req.max_tokens,
                "total_tokens": estimate_prompt_words(req.messages) + req.max_tokens,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
