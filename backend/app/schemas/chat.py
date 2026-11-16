"""Pydantic v2 contracts for chat SSE streaming, readiness probes, and telemetry."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agents.validators import CitationItem


class ChatStreamRequest(BaseModel):
    """Input payload for POST /api/v1/chat/stream."""

    model_config = ConfigDict(frozen=True)

    collection_id: str = Field(min_length=2, max_length=64)
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    inference_mode: Literal["LOCAL_ONPREM", "CLOUD_API"] | None = None
    language_filter: Literal["en", "ms", "mixed"] | None = None
    doc_type_filter: Literal["pdf", "md", "txt"] | None = None


class NodeStatusEventPayload(BaseModel):
    """Payload for SSE `event: node_status`."""

    model_config = ConfigDict(frozen=True)

    node: str
    status: str
    attempt: int
    latency_ms: int
    detail: str = ""


class TokenEventPayload(BaseModel):
    """Payload for SSE `event: token`."""

    model_config = ConfigDict(frozen=True)

    delta: str


class CitationsEventPayload(BaseModel):
    """Payload for SSE `event: citations`."""

    model_config = ConfigDict(frozen=True)

    sources: list[CitationItem]


class DoneEventPayload(BaseModel):
    """Payload for SSE `event: done`."""

    model_config = ConfigDict(frozen=True)

    message_id: str
    session_id: str
    mode: Literal["LOCAL_ONPREM", "CLOUD_API"]
    retries_used: int
    cache_hit_tokens: int
    total_latency_ms: int
    refusal_triggered: bool = False


class ErrorEventPayload(BaseModel):
    """Payload for SSE `event: error` emitted when streaming fails after HTTP 200."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    recoverable: bool = True


class ReadinessResponse(BaseModel):
    """Response model for GET /ready infrastructure health verification."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ready", "degraded"]
    checks: dict[str, str]
