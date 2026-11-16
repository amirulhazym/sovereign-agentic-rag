"""RAGTargetAdapter seam decoupling eval/ judges from FastAPI/backend internals."""

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

from eval_harness.schemas import EvalSample, RAGOutput


class RAGTargetAdapter(ABC):
    """Seam allowing judges to benchmark live HTTP RAG servers or recorded JSONL traces."""

    @abstractmethod
    async def evaluate_sample(self, sample: EvalSample) -> RAGOutput:
        """Query the target RAG system for one sample and normalize its output."""


@dataclass
class _SSEAccumulator:
    tokens: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    retrieved_contexts: list[str] = field(default_factory=list)
    refusal_triggered: bool = False
    cache_hit_tokens: int = 0
    reported_latency_ms: float | None = None

    def apply_block(self, block: str) -> None:
        parsed = _parse_sse_block(block)
        if parsed is None:
            return
        event_name, payload = parsed
        if event_name == "token":
            delta = str(payload.get("delta", ""))
            if delta:
                self.tokens.append(delta)
        elif event_name == "citations":
            for source in payload.get("sources", []):
                key = str(source.get("citation_key", "")).strip()
                if key:
                    self.citations.append(key)
                context_text = str(
                    source.get("snippet")
                    or source.get("excerpt")
                    or source.get("content")
                    or ""
                ).strip()
                if context_text:
                    self.retrieved_contexts.append(context_text)
        elif event_name == "done":
            self.refusal_triggered = bool(payload.get("refusal_triggered", False))
            self.cache_hit_tokens = int(payload.get("cache_hit_tokens", 0) or 0)
            if "total_latency_ms" in payload:
                self.reported_latency_ms = float(payload["total_latency_ms"])
        elif event_name == "error":
            code = payload.get("code", "SSE_ERROR")
            message = payload.get("message", "Unknown streaming failure")
            raise RuntimeError(f"Target SSE error [{code}]: {message}")


def _parse_sse_block(block: str) -> tuple[str, dict[str, Any]] | None:
    """Parse one complete SSE event block separated by blank lines."""
    event_name = "message"
    data_lines: list[str] = []
    for raw_line in block.splitlines():
        if not raw_line or raw_line.startswith(":"):
            continue
        if raw_line.startswith("event:"):
            event_name = raw_line[len("event:") :].strip()
        elif raw_line.startswith("data:"):
            data_lines.append(raw_line[len("data:") :].strip())
    if not data_lines:
        return None
    payload = json.loads("\n".join(data_lines))
    return event_name, payload


class HTTPTargetAdapter(RAGTargetAdapter):
    """Target adapter consuming FastAPI POST /api/v1/chat/stream over SSE."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000",
        inference_mode: Literal["LOCAL_ONPREM", "CLOUD_API"] | None = None,
        timeout_seconds: float = 120.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._inference_mode = inference_mode
        self._timeout = timeout_seconds
        self._injected_client = http_client

    async def evaluate_sample(self, sample: EvalSample) -> RAGOutput:
        """Stream `/api/v1/chat/stream` and assemble a normalized RAGOutput."""
        request_body: dict[str, Any] = {
            "collection_id": sample.collection,
            "question": sample.question,
        }
        if self._inference_mode is not None:
            request_body["inference_mode"] = self._inference_mode

        if self._injected_client is not None:
            return await self._execute_stream(self._injected_client, sample, request_body)

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        ) as client:
            return await self._execute_stream(client, sample, request_body)

    async def _execute_stream(
        self,
        client: httpx.AsyncClient,
        sample: EvalSample,
        request_body: dict[str, Any],
    ) -> RAGOutput:
        started_at = time.perf_counter()
        acc = _SSEAccumulator()

        async with client.stream(
            "POST",
            f"{self._base_url}/api/v1/chat/stream",
            json=request_body,
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status_code != 200:
                body_text = (await response.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Target API returned HTTP {response.status_code} for {sample.id}: {body_text}"
                )

            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk.replace("\r\n", "\n")
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    acc.apply_block(block)

            if buffer.strip():
                acc.apply_block(buffer)

        wall_ms = (time.perf_counter() - started_at) * 1000.0
        return RAGOutput(
            sample_id=sample.id,
            answer="".join(acc.tokens).strip(),
            retrieved_contexts=acc.retrieved_contexts,
            citations=acc.citations,
            refusal_triggered=acc.refusal_triggered,
            target_latency_ms=(
                acc.reported_latency_ms
                if acc.reported_latency_ms is not None
                else round(wall_ms, 2)
            ),
            cache_hit_tokens=acc.cache_hit_tokens,
        )


class RecordedTraceTargetAdapter(RAGTargetAdapter):
    """Offline target adapter replaying recorded RAGOutput JSONL traces for deterministic CI."""

    def __init__(self, outputs: Iterable[RAGOutput]) -> None:
        self._outputs_by_id: dict[str, RAGOutput] = {out.sample_id: out for out in outputs}

    @classmethod
    def from_jsonl(cls, path: Path) -> "RecordedTraceTargetAdapter":
        """Load recorded RAGOutput entries from a JSONL trace file."""
        if not path.is_file():
            raise FileNotFoundError(f"Recorded trace JSONL not found: {path}")
        records: list[RAGOutput] = []
        for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                records.append(RAGOutput.model_validate_json(stripped))
            except Exception as exc:
                raise ValueError(f"Invalid RAGOutput on line {line_no} of {path}: {exc}") from exc
        return cls(records)

    async def evaluate_sample(self, sample: EvalSample) -> RAGOutput:
        """Return previously captured RAGOutput for the matching sample ID."""
        if sample.id not in self._outputs_by_id:
            raise KeyError(f"No recorded RAGOutput trace found for sample_id='{sample.id}'")
        return self._outputs_by_id[sample.id]
