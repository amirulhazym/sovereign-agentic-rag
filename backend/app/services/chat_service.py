"""Orchestrates chat session history, dual-mode provider selection, and typed SSE streaming."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from app.agents.graph import AgenticRAGWorkflow
from app.core.exceptions import AppError, ProviderUnavailableError
from app.providers.base import ChatMessage, LLMProvider
from app.providers.embeddings import EmbeddingProvider, RerankerProvider
from app.repositories.qdrant_repo import VectorRepository
from app.repositories.sqlite_repo import SQLiteRepository
from app.schemas.chat import (
    ChatStreamRequest,
    CitationsEventPayload,
    DoneEventPayload,
    ErrorEventPayload,
    NodeStatusEventPayload,
    TokenEventPayload,
)
from app.schemas.workspace import CollectionRead

logger = logging.getLogger(__name__)


def format_sse(event_name: str, payload: BaseModel) -> str:
    """Serialize a typed Pydantic payload into a standard SSE wire frame."""
    return f"event: {event_name}\ndata: {payload.model_dump_json()}\n\n"


class ChatService:
    """Deep service handling conversation state, workflow execution, and SSE streaming."""

    def __init__(
        self,
        *,
        sqlite_repo: SQLiteRepository,
        vector_repo: VectorRepository,
        embedding_provider: EmbeddingProvider,
        reranker_provider: RerankerProvider,
        llm_providers: dict[str, LLMProvider],
        default_inference_mode: Literal["LOCAL_ONPREM", "CLOUD_API"] = "LOCAL_ONPREM",
        max_retries: int = 2,
    ) -> None:
        self._sqlite = sqlite_repo
        self._vectors = vector_repo
        self._embedder = embedding_provider
        self._reranker = reranker_provider
        self._llm_providers = llm_providers
        self._default_mode = default_inference_mode
        self._max_retries = max_retries

    async def prepare_stream(
        self, request: ChatStreamRequest
    ) -> tuple[CollectionRead, str, list[ChatMessage], Literal["LOCAL_ONPREM", "CLOUD_API"]]:
        """Validate collection & provider, load prior history, and record user message."""
        collection = await self._sqlite.get_collection(request.collection_id)
        mode: Literal["LOCAL_ONPREM", "CLOUD_API"] = (
            request.inference_mode or self._default_mode
        )
        if mode not in self._llm_providers:
            raise ProviderUnavailableError(
                "PROVIDER_NOT_CONFIGURED",
                f"Inference mode '{mode}' is not configured on this server.",
            )

        session_id = request.session_id or f"sess_{uuid4().hex[:12]}"
        await self._sqlite.ensure_chat_session(
            session_id=session_id,
            collection_id=collection.id,
            title=request.question[:60],
        )

        # Load prior turns BEFORE inserting the current user question to avoid duplicate tail
        prior_records = await self._sqlite.list_session_messages(session_id)
        chat_history = [
            ChatMessage(role=msg.role, content=msg.content) for msg in prior_records
        ]

        await self._sqlite.append_chat_message(
            session_id=session_id,
            role="user",
            content=request.question,
            inference_mode=mode,
        )
        return collection, session_id, chat_history, mode

    async def stream_events(
        self,
        *,
        request: ChatStreamRequest,
        collection: CollectionRead,
        session_id: str,
        chat_history: list[ChatMessage],
        mode: Literal["LOCAL_ONPREM", "CLOUD_API"],
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[str]:
        """Execute AgenticRAGWorkflow and yield typed SSE frames cleanly."""
        llm_provider = self._llm_providers[mode]
        workflow = AgenticRAGWorkflow(
            sqlite_repo=self._sqlite,
            vector_repo=self._vectors,
            embedding_provider=self._embedder,
            reranker_provider=self._reranker,
            llm_provider=llm_provider,
            max_retries=self._max_retries,
        )
        agent_trace: list[dict[str, object]] = []

        try:
            async for event in workflow.astream_run(
                question=request.question,
                collection_id=collection.id,
                collection_description=collection.description,
                chat_history=chat_history,
                inference_mode=mode,
                language_filter=request.language_filter,
                doc_type_filter=request.doc_type_filter,
            ):
                if is_disconnected is not None and await is_disconnected():
                    logger.info("SSE client disconnected for session_id=%s", session_id)
                    break

                if event.event_type == "node_status":
                    payload = NodeStatusEventPayload(
                        node=event.node,
                        status=event.status,
                        attempt=event.attempt,
                        latency_ms=event.latency_ms,
                        detail=event.detail,
                    )
                    if event.status in {"completed", "retrying"}:
                        agent_trace.append(payload.model_dump())
                    yield format_sse("node_status", payload)
                elif event.event_type == "token":
                    yield format_sse("token", TokenEventPayload(delta=event.delta))
                elif event.event_type == "citations":
                    yield format_sse(
                        "citations",
                        CitationsEventPayload(sources=event.citations),
                    )
                elif event.event_type == "done":
                    message_id = f"msg_{uuid4().hex[:12]}"
                    await self._sqlite.append_chat_message(
                        message_id=message_id,
                        session_id=session_id,
                        role="assistant",
                        content=event.answer,
                        inference_mode=mode,
                        citations=[c.model_dump() for c in event.citations],
                        agent_trace=agent_trace,
                        cache_hit_tokens=event.cache_hit_tokens,
                        latency_ms=event.latency_ms,
                    )
                    yield format_sse(
                        "done",
                        DoneEventPayload(
                            message_id=message_id,
                            session_id=session_id,
                            mode=mode,
                            retries_used=event.retries_used,
                            cache_hit_tokens=event.cache_hit_tokens,
                            total_latency_ms=event.latency_ms,
                            refusal_triggered=event.refusal_triggered,
                        ),
                    )
        except AppError as exc:
            yield format_sse(
                "error",
                ErrorEventPayload(
                    code=exc.code,
                    message=exc.message,
                    recoverable=True,
                ),
            )
        except Exception:
            logger.exception("Unexpected SSE stream failure for session_id=%s", session_id)
            yield format_sse(
                "error",
                ErrorEventPayload(
                    code="STREAM_INTERNAL_ERROR",
                    message="An unexpected error occurred while streaming the response.",
                    recoverable=False,
                ),
            )
