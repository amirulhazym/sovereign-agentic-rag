"""Cloud DeepSeek API adapter optimized for Prefix Context Caching telemetry."""

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


class CloudDeepSeekLLMAdapter:
    """OpenAI-compatible DeepSeek adapter enforcing JSON control mode and cache telemetry."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        model: str = "deepseek-chat",
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._model = model

    async def ping(self) -> bool:
        if not self._api_key:
            return False
        try:
            resp = await self._client.get(
                "/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            return resp.status_code < 500
        except httpx.HTTPError:
            return False

    async def generate_structured(
        self,
        messages: list[ChatMessage],
        schema: type[TModel],
    ) -> TModel:
        """Run temperature=0.0 JSON-mode control decision on DeepSeek."""
        if not self._api_key:
            raise ProviderUnavailableError(
                "DEEPSEEK_API_KEY_MISSING",
                "DEEPSEEK_API_KEY is required for CLOUD_API mode.",
            )
        try:
            resp = await self._client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [m.model_dump() for m in messages],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                    "stream": False,
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                "DEEPSEEK_UNAVAILABLE",
                f"DeepSeek API request failed: {exc}",
            ) from exc

        content = resp.json()["choices"][0]["message"]["content"]
        return schema.model_validate(extract_json_object(content))

    async def stream_answer(
        self,
        messages: list[ChatMessage],
    ) -> AsyncIterator[StreamChunk]:
        """Stream DeepSeek completion tokens and capture prompt_cache_hit_tokens."""
        if not self._api_key:
            raise ProviderUnavailableError(
                "DEEPSEEK_API_KEY_MISSING",
                "DEEPSEEK_API_KEY is required for CLOUD_API mode.",
            )
        cache_hit_tokens = 0
        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [m.model_dump() for m in messages],
                    "temperature": 0.2,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            ) as resp:
                resp.raise_for_status()
                async for event in iter_openai_sse_events(resp):
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        cache_hit_tokens = int(usage.get("prompt_cache_hit_tokens", 0))
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}).get("content", "")
                    if delta:
                        yield StreamChunk(delta=delta)
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                "DEEPSEEK_UNAVAILABLE",
                f"DeepSeek streaming failed: {exc}",
            ) from exc

        yield StreamChunk(delta="", is_final=True, cache_hit_tokens=cache_hit_tokens)
