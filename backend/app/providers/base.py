"""LLMProvider seam, message contracts, shared SSE parser, and deterministic fake."""

import json
import re
from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict

TModel = TypeVar("TModel", bound=BaseModel)


class ChatMessage(BaseModel):
    """Single prompt message ordered static-to-dynamic for prefix caching."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class StreamChunk(BaseModel):
    """Incremental text delta or final usage telemetry yielded by LLMProvider."""

    model_config = ConfigDict(frozen=True)

    delta: str = ""
    is_final: bool = False
    cache_hit_tokens: int = 0


class LLMProvider(Protocol):
    """Seam swapping LOCAL_ONPREM (Ollama Qwen3.8) and CLOUD_API (DeepSeek)."""

    async def ping(self) -> bool: ...

    async def generate_structured(
        self,
        messages: list[ChatMessage],
        schema: type[TModel],
    ) -> TModel: ...

    def stream_answer(
        self,
        messages: list[ChatMessage],
    ) -> AsyncIterator[StreamChunk]: ...


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse JSON from model output, tolerating markdown code fences."""
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(0)
    return dict(json.loads(cleaned))


async def iter_openai_sse_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Yield decoded JSON objects from an OpenAI-compatible SSE chat completion stream."""
    async for line in response.aiter_lines():
        stripped = line.strip()
        if not stripped or not stripped.startswith("data:"):
            continue
        payload = stripped.removeprefix("data:").strip()
        if payload == "[DONE]":
            break
        yield dict(json.loads(payload))


class FakeLLMProvider:
    """In-memory deterministic LLMProvider used by LangGraph and SSE unit tests."""

    def __init__(
        self,
        structured_responses: list[BaseModel] | None = None,
        stream_tokens: list[str] | None = None,
        cache_hit_tokens: int = 0,
    ) -> None:
        self._structured_queue = list(structured_responses or [])
        self._stream_tokens = list(stream_tokens or ["Sample answer."])
        self._cache_hit_tokens = cache_hit_tokens
        self.calls: list[list[ChatMessage]] = []

    async def ping(self) -> bool:
        return True

    async def generate_structured(
        self,
        messages: list[ChatMessage],
        schema: type[TModel],
    ) -> TModel:
        self.calls.append(messages)
        if not self._structured_queue:
            raise RuntimeError("FakeLLMProvider structured_responses queue exhausted.")
        item = self._structured_queue.pop(0)
        return schema.model_validate(item.model_dump())

    async def stream_answer(
        self,
        messages: list[ChatMessage],
    ) -> AsyncIterator[StreamChunk]:
        self.calls.append(messages)
        for token in self._stream_tokens:
            yield StreamChunk(delta=token)
        yield StreamChunk(
            delta="",
            is_final=True,
            cache_hit_tokens=self._cache_hit_tokens,
        )
