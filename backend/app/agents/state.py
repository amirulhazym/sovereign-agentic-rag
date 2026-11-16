"""LangGraph state schema, control-node JSON contracts, and stream event envelope."""

from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from app.agents.validators import CitationItem
from app.providers.base import ChatMessage
from app.providers.embeddings import ScoredParent
from app.schemas.workspace import ParentChunkRecord


class RouterOutput(BaseModel):
    """Structured JSON output emitted by QueryRouter (/no_think, temperature=0.0)."""

    model_config = ConfigDict(frozen=True)

    route: Literal["direct", "single_collection", "multi_doc_compare"]
    detected_language: Literal["en", "ms", "mixed"]


class RewriterOutput(BaseModel):
    """Structured JSON output emitted by QueryRewriter (/no_think, temperature=0.0)."""

    model_config = ConfigDict(frozen=True)

    queries: list[str]
    reasoning: str = ""


class GraderOutput(BaseModel):
    """Structured JSON output emitted by RelevanceGrader (/no_think, temperature=0.0)."""

    model_config = ConfigDict(frozen=True)

    is_sufficient: bool
    approved_parent_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class WorkflowEvent(BaseModel):
    """Unified streaming event produced by AgenticRAGWorkflow during graph execution."""

    model_config = ConfigDict(frozen=True)

    event_type: Literal["node_status", "token", "citations", "done"]
    node: str = ""
    status: str = ""
    attempt: int = 0
    latency_ms: int = 0
    detail: str = ""
    delta: str = ""
    citations: list[CitationItem] = Field(default_factory=list)
    answer: str = ""
    retries_used: int = 0
    cache_hit_tokens: int = 0
    refusal_triggered: bool = False


class RAGGraphState(TypedDict):
    """Centralized state passed across all nodes of the bounded LangGraph workflow."""

    question: str
    collection_id: str
    collection_description: str
    chat_history: list[ChatMessage]
    inference_mode: Literal["LOCAL_ONPREM", "CLOUD_API"]
    language_filter: str | None
    doc_type_filter: str | None
    detected_language: Literal["en", "ms", "mixed"]
    route: Literal["direct", "single_collection", "multi_doc_compare"]
    rewritten_queries: list[str]
    previous_queries: list[str]
    candidate_parents: list[ParentChunkRecord]
    reranked_parents: list[ScoredParent]
    relevant_parents: list[ScoredParent]
    retry_count: int
    answer: str
    validated_citations: list[CitationItem]
    refusal_triggered: bool
    cache_hit_tokens: int
