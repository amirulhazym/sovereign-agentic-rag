"""Local on-premises LLM adapter targeting Ollama / llama.cpp (Qwen3.8-Distill series).

Handles `<think>...</think>` reasoning spans emitted by `empero-ai/Qwen3.8-*-Distill-GGUF`
both in non-streaming structured control nodes and across split streaming token chunks.
"""

import re
from collections.abc import AsyncIterator
from typing import TypeVar

import httpx
from pydantic import BaseModel

from app.core.exceptions import ProviderUnavailableError
from app.providers.base import (
    ChatMessage,
    StreamChunk,
    extract_json_object,
    iter_openai_sse_events,
)

TModel = TypeVar("TModel", bound=BaseModel)

_OPEN_THINK = "<think>"
_CLOSE_THINK = "</think>"


def strip_thinking_blocks(text: str) -> str:
    """Remove complete <think>...</think> spans and leading whitespace from model output."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.lstrip()


class StreamingThinkFilter:
    """Stateful filter stripping <think>...</think> spans even across arbitrary token splits."""

    def __init__(self) -> None:
        self._in_think = False
        self._buffer = ""
        self._emitted_any = False

    def feed(self, token: str) -> str:
        """Consume a raw stream delta and return only safe non-thinking text."""
        self._buffer += token
        output_parts: list[str] = []

        while self._buffer:
            if self._in_think:
                end_idx = self._buffer.find(_CLOSE_THINK)
                if end_idx == -1:
                    # Retain only enough tail chars to catch a split closing tag across chunks
                    keep_tail = len(_CLOSE_THINK) - 1
                    if len(self._buffer) > keep_tail:
                        self._buffer = self._buffer[-keep_tail:]
                    break
                self._buffer = self._buffer[end_idx + len(_CLOSE_THINK) :]
                self._in_think = False
                if not self._emitted_any:
                    self._buffer = self._buffer.lstrip()
                continue

            start_idx = self._buffer.find(_OPEN_THINK)
            if start_idx != -1:
                prefix = self._buffer[:start_idx]
                if prefix:
                    self._emitted_any = True
                    output_parts.append(prefix)
                self._buffer = self._buffer[start_idx + len(_OPEN_THINK) :]
                self._in_think = True
                continue

            # Hold trailing characters if they match a partial `<think>` prefix
            hold_len = 0
            max_check = min(len(self._buffer), len(_OPEN_THINK) - 1)
            for suffix_len in range(max_check, 0, -1):
                if _OPEN_THINK.startswith(self._buffer[-suffix_len:]):
                    hold_len = suffix_len
                    break

            emit_end = len(self._buffer) - hold_len
            if emit_end > 0:
                chunk = self._buffer[:emit_end]
                if not self._emitted_any:
                    chunk = chunk.lstrip()
                if chunk:
                    self._emitted_any = True
                    output_parts.append(chunk)
                self._buffer = self._buffer[emit_end:]
            break

        return "".join(output_parts)

    def flush(self) -> str:
        """Flush any buffered non-thinking text when the stream finishes."""
        if self._in_think or not self._buffer:
            self._buffer = ""
            return ""
        tail = self._buffer if self._emitted_any else self._buffer.lstrip()
        self._buffer = ""
        return tail


class LocalOnPremLLMAdapter:
    """Adapter calling local Ollama/llama.cpp OpenAI-compatible chat completions."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        model: str = "hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M",
        api_key: str = "ollama",
    ) -> None:
        self._client = client
        self._model = model
        self._api_key = api_key

    async def ping(self) -> bool:
        try:
            resp = await self._client.get("/models")
            return resp.status_code < 500
        except httpx.HTTPError:
            return False

    async def generate_structured(
        self,
        messages: list[ChatMessage],
        schema: type[TModel],
    ) -> TModel:
        """Run deterministic temperature=0.0 JSON control decision with /no_think hint."""
        payload_messages = [m.model_dump() for m in messages]
        if payload_messages:
            payload_messages[-1] = {
                "role": payload_messages[-1]["role"],
                "content": f"{payload_messages[-1]['content']}\n/no_think",
            }
        try:
            resp = await self._client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": payload_messages,
                    "temperature": 0.0,
                    "stream": False,
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                "OLLAMA_UNAVAILABLE",
                f"Local Ollama inference error: {exc}",
            ) from exc

        raw_content = resp.json()["choices"][0]["message"]["content"]
        cleaned = strip_thinking_blocks(raw_content)
        return schema.model_validate(extract_json_object(cleaned))

    async def stream_answer(
        self,
        messages: list[ChatMessage],
    ) -> AsyncIterator[StreamChunk]:
        """Stream answer tokens with Qwen3.8 recommended sampling and <think> stripping."""
        think_filter = StreamingThinkFilter()
        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [m.model_dump() for m in messages],
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "stream": True,
                },
            ) as resp:
                resp.raise_for_status()
                async for event in iter_openai_sse_events(resp):
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}).get("content", "")
                    if delta:
                        safe_text = think_filter.feed(delta)
                        if safe_text:
                            yield StreamChunk(delta=safe_text)
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                "OLLAMA_UNAVAILABLE",
                f"Local Ollama streaming error: {exc}",
            ) from exc

        tail = think_filter.flush()
        if tail:
            yield StreamChunk(delta=tail)
        yield StreamChunk(delta="", is_final=True, cache_hit_tokens=0)
