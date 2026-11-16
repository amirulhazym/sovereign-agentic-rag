"""Six observable LangGraph nodes plus deterministic GracefulRefusal terminal node."""

import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.state import (
    GraderOutput,
    RAGGraphState,
    RewriterOutput,
    RouterOutput,
    WorkflowEvent,
)
from app.agents.validators import (
    build_graceful_refusal,
    validate_and_normalize_citations,
)
from app.providers.base import ChatMessage, LLMProvider
from app.providers.embeddings import EmbeddingProvider, RerankerProvider
from app.providers.prompts import build_grounded_messages
from app.repositories.qdrant_repo import VectorRepository
from app.repositories.sqlite_repo import SQLiteRepository

EventEmitter = Callable[[WorkflowEvent], Awaitable[None]]


async def _emit(config: RunnableConfig, event: WorkflowEvent) -> None:
    emitter: EventEmitter | None = config.get("configurable", {}).get("emitter")
    if emitter is not None:
        await emitter(event)


def _elapsed_ms(start: float) -> int:
    return max(int((time.perf_counter() - start) * 1000), 0)


class RAGAgentNodes:
    """Cohesive node implementations bound to injected storage and provider seams."""

    def __init__(
        self,
        *,
        sqlite_repo: SQLiteRepository,
        vector_repo: VectorRepository,
        embedding_provider: EmbeddingProvider,
        reranker_provider: RerankerProvider,
        llm_provider: LLMProvider,
        max_retries: int = 2,
    ) -> None:
        self._sqlite = sqlite_repo
        self._vectors = vector_repo
        self._embedder = embedding_provider
        self._reranker = reranker_provider
        self._llm = llm_provider
        self._max_retries = max_retries

    async def query_router(
        self, state: RAGGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Classify query intent ('direct' vs retrieval) and detect EN/BM/mixed language."""
        start = time.perf_counter()
        attempt = state.get("retry_count", 0)
        await _emit(
            config,
            WorkflowEvent(
                event_type="node_status",
                node="QueryRouter",
                status="started",
                attempt=attempt,
            ),
        )
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Classify the user message for an enterprise knowledge workspace.\n"
                    "Return JSON with keys `route` and `detected_language`.\n"
                    "- `route`: 'direct' ONLY for greetings/thanks/chit-chat requiring no "
                    "documents; 'single_collection' for normal factual/SOP/policy queries; "
                    "'multi_doc_compare' when comparing multiple standards, tiers, or policies.\n"
                    "- `detected_language`: 'en', 'ms', or 'mixed' (BM-EN code-switching)."
                ),
            ),
            ChatMessage(role="user", content=state["question"]),
        ]
        decision = await self._llm.generate_structured(messages, RouterOutput)
        await _emit(
            config,
            WorkflowEvent(
                event_type="node_status",
                node="QueryRouter",
                status="completed",
                attempt=attempt,
                latency_ms=_elapsed_ms(start),
                detail=f"route={decision.route} lang={decision.detected_language}",
            ),
        )
        return {
            "route": decision.route,
            "detected_language": decision.detected_language,
        }

    async def query_rewriter(
        self, state: RAGGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Expand technical acronyms and bilingual BM<->EN search terms."""
        start = time.perf_counter()
        attempt = state.get("retry_count", 0)
        await _emit(
            config,
            WorkflowEvent(
                event_type="node_status",
                node="QueryRewriter",
                status="started",
                attempt=attempt,
            ),
        )
        previous = state.get("previous_queries", [])
        retry_hint = ""
        if attempt > 0 and previous:
            retry_hint = (
                f"\nPrevious queries failed relevance check: {previous}. "
                "Generate alternative terminology, synonyms, or cross-language BM<->EN equivalents."
            )
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Rewrite the user question into 1-2 concise hybrid search queries.\n"
                    "Expand acronyms (e.g. OT -> overtime kerja lebih masa, DR -> disaster "
                    "recovery RPO RTO) and bridge Bahasa Melayu <-> English terms.\n"
                    "Return JSON: {\"queries\": [\"...\"], \"reasoning\": \"...\"}."
                ),
            ),
            ChatMessage(
                role="user",
                content=f"Question: {state['question']}{retry_hint}",
            ),
        ]
        rewritten = await self._llm.generate_structured(messages, RewriterOutput)
        queries = [q.strip() for q in rewritten.queries if q.strip()] or [state["question"]]
        updated_previous = list(dict.fromkeys([*previous, *queries]))
        await _emit(
            config,
            WorkflowEvent(
                event_type="node_status",
                node="QueryRewriter",
                status="completed",
                attempt=attempt,
                latency_ms=_elapsed_ms(start),
                detail=f"queries={len(queries)}",
            ),
        )
        return {
            "rewritten_queries": queries,
            "previous_queries": updated_previous,
        }

    async def hybrid_retriever(
        self, state: RAGGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Run Dense+Sparse Qdrant RRF scoped by collection_id and hydrate ParentChunks."""
        start = time.perf_counter()
        attempt = state.get("retry_count", 0)
        await _emit(
            config,
            WorkflowEvent(
                event_type="node_status",
                node="HybridRetriever",
                status="started",
                attempt=attempt,
            ),
        )
        queries = state.get("rewritten_queries") or [state["question"]]
        dense_vecs = await self._embedder.embed_dense(queries)
        sparse_vecs = await self._embedder.embed_sparse(queries)

        parent_ids: list[str] = []
        total_child_hits = 0
        for dense_vec, sparse_vec in zip(dense_vecs, sparse_vecs, strict=True):
            hits = await self._vectors.hybrid_search(
                collection_id=state["collection_id"],
                dense_vector=dense_vec,
                sparse_vector=sparse_vec,
                top_k=12,
                language=state.get("language_filter"),
                doc_type=state.get("doc_type_filter"),
            )
            total_child_hits += len(hits)
            for hit in hits:
                if hit.parent_id not in parent_ids:
                    parent_ids.append(hit.parent_id)

        hydrated_parents = await self._sqlite.get_parent_chunks_by_ids(parent_ids)
        await _emit(
            config,
            WorkflowEvent(
                event_type="node_status",
                node="HybridRetriever",
                status="completed",
                attempt=attempt,
                latency_ms=_elapsed_ms(start),
                detail=f"{total_child_hits} child hits -> {len(hydrated_parents)} unique parents",
            ),
        )
        return {"candidate_parents": hydrated_parents}

    async def cross_encoder_reranker(
        self, state: RAGGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Re-score hydrated ParentChunks with the Cross-Encoder reranker (Top-5)."""
        start = time.perf_counter()
        attempt = state.get("retry_count", 0)
        await _emit(
            config,
            WorkflowEvent(
                event_type="node_status",
                node="CrossEncoderReranker",
                status="started",
                attempt=attempt,
            ),
        )
        candidates = state.get("candidate_parents", [])
        reranked = await self._reranker.rerank(state["question"], candidates, top_k=5)
        await _emit(
            config,
            WorkflowEvent(
                event_type="node_status",
                node="CrossEncoderReranker",
                status="completed",
                attempt=attempt,
                latency_ms=_elapsed_ms(start),
                detail=f"top {len(reranked)} scored",
            ),
        )
        return {"reranked_parents": reranked}

    async def relevance_grader(
        self, state: RAGGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Self-reflective evidence check enforcing the bounded max_retries=2 loop cap."""
        start = time.perf_counter()
        attempt = state.get("retry_count", 0)
        await _emit(
            config,
            WorkflowEvent(
                event_type="node_status",
                node="RelevanceGrader",
                status="started",
                attempt=attempt,
            ),
        )
        reranked = state.get("reranked_parents", [])

        if not reranked:
            next_retry = attempt + 1 if attempt < self._max_retries else attempt
            status_label = "retrying" if attempt < self._max_retries else "completed"
            await _emit(
                config,
                WorkflowEvent(
                    event_type="node_status",
                    node="RelevanceGrader",
                    status=status_label,
                    attempt=attempt,
                    latency_ms=_elapsed_ms(start),
                    detail="0 candidates found",
                ),
            )
            return {
                "relevant_parents": [],
                "retry_count": next_retry,
                "refusal_triggered": attempt >= self._max_retries,
            }

        evidence_preview = "\n\n".join(
            f"[parent_id={item.parent.parent_id}] ({item.parent.breadcrumb})\n"
            f"{item.parent.content[:800]}"
            for item in reranked
        )
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a strict relevance grader. Determine whether the candidate passages "
                    "contain sufficient factual evidence to answer the user question.\n"
                    "Return JSON: {\"is_sufficient\": bool, \"approved_parent_ids\": [\"...\"], "
                    "\"reason\": \"...\"}."
                ),
            ),
            ChatMessage(
                role="user",
                content=f"Question: {state['question']}\n\nCandidates:\n{evidence_preview}",
            ),
        ]
        verdict = await self._llm.generate_structured(messages, GraderOutput)

        if verdict.is_sufficient:
            approved_set = set(verdict.approved_parent_ids)
            selected = [item for item in reranked if item.parent.parent_id in approved_set]
            if not selected:
                selected = reranked
            await _emit(
                config,
                WorkflowEvent(
                    event_type="node_status",
                    node="RelevanceGrader",
                    status="completed",
                    attempt=attempt,
                    latency_ms=_elapsed_ms(start),
                    detail=f"{len(selected)} relevant parents approved",
                ),
            )
            return {
                "relevant_parents": selected,
                "refusal_triggered": False,
            }

        will_retry = attempt < self._max_retries
        next_retry = attempt + 1 if will_retry else attempt
        await _emit(
            config,
            WorkflowEvent(
                event_type="node_status",
                node="RelevanceGrader",
                status="retrying" if will_retry else "completed",
                attempt=attempt,
                latency_ms=_elapsed_ms(start),
                detail=verdict.reason or "Insufficient evidence",
            ),
        )
        return {
            "relevant_parents": [],
            "retry_count": next_retry,
            "refusal_triggered": not will_retry,
        }

    async def grounded_generator(
        self, state: RAGGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Stream grounded answer tokens and validate inline [doc_id:pX] citations."""
        start = time.perf_counter()
        attempt = state.get("retry_count", 0)
        await _emit(
            config,
            WorkflowEvent(
                event_type="node_status",
                node="GroundedGenerator",
                status="started",
                attempt=attempt,
            ),
        )

        relevant_scored = state.get("relevant_parents", [])
        if state.get("route") == "direct":
            messages = [
                ChatMessage(
                    role="system",
                    content=(
                        "You are Sovereign Agentic RAG. Respond politely and concisely in the "
                        "user's language to their greeting or meta-question."
                    ),
                ),
                *state.get("chat_history", []),
                ChatMessage(role="user", content=state["question"]),
            ]
        else:
            messages = build_grounded_messages(
                collection_id=state["collection_id"],
                collection_description=state.get("collection_description", ""),
                history=state.get("chat_history", []),
                parents=[item.parent for item in relevant_scored],
                question=state["question"],
            )

        raw_parts: list[str] = []
        cache_hit_tokens = 0
        async for chunk in self._llm.stream_answer(messages):
            if chunk.delta:
                raw_parts.append(chunk.delta)
                await _emit(config, WorkflowEvent(event_type="token", delta=chunk.delta))
            if chunk.is_final:
                cache_hit_tokens = chunk.cache_hit_tokens

        raw_answer = "".join(raw_parts)
        validated_answer, citations = validate_and_normalize_citations(
            raw_answer, relevant_scored
        )
        await _emit(config, WorkflowEvent(event_type="citations", citations=citations))
        await _emit(
            config,
            WorkflowEvent(
                event_type="node_status",
                node="GroundedGenerator",
                status="completed",
                attempt=attempt,
                latency_ms=_elapsed_ms(start),
                detail=f"citations={len(citations)}",
            ),
        )
        return {
            "answer": validated_answer,
            "validated_citations": citations,
            "cache_hit_tokens": cache_hit_tokens,
            "refusal_triggered": False,
        }

    async def graceful_refusal(
        self, state: RAGGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Emit deterministic bilingual refusal with zero citations when retries exhaust."""
        start = time.perf_counter()
        attempt = state.get("retry_count", 0)
        await _emit(
            config,
            WorkflowEvent(
                event_type="node_status",
                node="GracefulRefusal",
                status="started",
                attempt=attempt,
            ),
        )
        refusal_text = build_graceful_refusal(state.get("detected_language", "mixed"))
        await _emit(config, WorkflowEvent(event_type="token", delta=refusal_text))
        await _emit(config, WorkflowEvent(event_type="citations", citations=[]))
        await _emit(
            config,
            WorkflowEvent(
                event_type="node_status",
                node="GracefulRefusal",
                status="completed",
                attempt=attempt,
                latency_ms=_elapsed_ms(start),
                detail="Bounded retries exhausted; emitted sovereign refusal",
            ),
        )
        return {
            "answer": refusal_text,
            "validated_citations": [],
            "refusal_triggered": True,
            "cache_hit_tokens": 0,
        }

    def route_after_router(self, state: RAGGraphState) -> str:
        if state.get("route") == "direct":
            return "GroundedGenerator"
        return "QueryRewriter"

    def route_after_grading(self, state: RAGGraphState) -> str:
        if state.get("relevant_parents"):
            return "GroundedGenerator"
        if state.get("refusal_triggered"):
            return "GracefulRefusal"
        return "QueryRewriter"
