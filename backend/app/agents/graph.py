"""Compiled bounded LangGraph StateGraph (max_retries=2) with async streaming runner."""

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from typing import Literal

from langgraph.graph import END, StateGraph

from app.agents.nodes import RAGAgentNodes
from app.agents.state import RAGGraphState, WorkflowEvent
from app.providers.base import ChatMessage, LLMProvider
from app.providers.embeddings import EmbeddingProvider, RerankerProvider
from app.repositories.qdrant_repo import VectorRepository
from app.repositories.sqlite_repo import SQLiteRepository


class AgenticRAGWorkflow:
    """Deep module compiling the 6-node LangGraph state machine and streaming its execution."""

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
        self._nodes = RAGAgentNodes(
            sqlite_repo=sqlite_repo,
            vector_repo=vector_repo,
            embedding_provider=embedding_provider,
            reranker_provider=reranker_provider,
            llm_provider=llm_provider,
            max_retries=max_retries,
        )
        self._graph = self._build_compiled_graph()

    def _build_compiled_graph(self):
        workflow = StateGraph(RAGGraphState)
        workflow.add_node("QueryRouter", self._nodes.query_router)
        workflow.add_node("QueryRewriter", self._nodes.query_rewriter)
        workflow.add_node("HybridRetriever", self._nodes.hybrid_retriever)
        workflow.add_node("CrossEncoderReranker", self._nodes.cross_encoder_reranker)
        workflow.add_node("RelevanceGrader", self._nodes.relevance_grader)
        workflow.add_node("GroundedGenerator", self._nodes.grounded_generator)
        workflow.add_node("GracefulRefusal", self._nodes.graceful_refusal)

        workflow.set_entry_point("QueryRouter")
        workflow.add_conditional_edges(
            "QueryRouter",
            self._nodes.route_after_router,
            {
                "GroundedGenerator": "GroundedGenerator",
                "QueryRewriter": "QueryRewriter",
            },
        )
        workflow.add_edge("QueryRewriter", "HybridRetriever")
        workflow.add_edge("HybridRetriever", "CrossEncoderReranker")
        workflow.add_edge("CrossEncoderReranker", "RelevanceGrader")
        workflow.add_conditional_edges(
            "RelevanceGrader",
            self._nodes.route_after_grading,
            {
                "GroundedGenerator": "GroundedGenerator",
                "QueryRewriter": "QueryRewriter",
                "GracefulRefusal": "GracefulRefusal",
            },
        )
        workflow.add_edge("GroundedGenerator", END)
        workflow.add_edge("GracefulRefusal", END)
        return workflow.compile()

    async def astream_run(
        self,
        *,
        question: str,
        collection_id: str,
        collection_description: str = "",
        chat_history: list[ChatMessage] | None = None,
        inference_mode: Literal["LOCAL_ONPREM", "CLOUD_API"] = "LOCAL_ONPREM",
        language_filter: str | None = None,
        doc_type_filter: str | None = None,
    ) -> AsyncIterator[WorkflowEvent]:
        """Execute LangGraph workflow and yield live node_status, token, citations, and done."""
        start_time = time.perf_counter()
        queue: asyncio.Queue[WorkflowEvent | Exception | None] = asyncio.Queue()
        initial_state: RAGGraphState = {
            "question": question,
            "collection_id": collection_id,
            "collection_description": collection_description,
            "chat_history": list(chat_history or []),
            "inference_mode": inference_mode,
            "language_filter": language_filter,
            "doc_type_filter": doc_type_filter,
            "detected_language": "mixed",
            "route": "single_collection",
            "rewritten_queries": [],
            "previous_queries": [],
            "candidate_parents": [],
            "reranked_parents": [],
            "relevant_parents": [],
            "retry_count": 0,
            "answer": "",
            "validated_citations": [],
            "refusal_triggered": False,
            "cache_hit_tokens": 0,
        }

        final_holder: dict[str, RAGGraphState] = {}

        async def _run_graph() -> None:
            try:
                final_state = await self._graph.ainvoke(
                    initial_state,
                    config={"configurable": {"emitter": queue.put}},
                )
                final_holder["state"] = final_state
            except Exception as exc:
                await queue.put(exc)
            finally:
                await queue.put(None)

        runner = asyncio.create_task(_run_graph())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item

            final_state = final_holder.get("state", initial_state)
            total_ms = max(int((time.perf_counter() - start_time) * 1000), 0)
            yield WorkflowEvent(
                event_type="done",
                answer=final_state.get("answer", ""),
                citations=final_state.get("validated_citations", []),
                retries_used=final_state.get("retry_count", 0),
                cache_hit_tokens=final_state.get("cache_hit_tokens", 0),
                latency_ms=total_ms,
                refusal_triggered=final_state.get("refusal_triggered", False),
            )
        finally:
            if not runner.done():
                runner.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await runner
