"""Pydantic v2 domain schemas for collections, documents, hierarchical chunks, and chat."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(StrEnum):
    """Ingestion lifecycle states persisted in SQLite and polled by Next.js."""

    PENDING = "PENDING"
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"


class CollectionCreate(BaseModel):
    """Input payload to register a new isolated tenant Collection."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9_]+$")
    name: str = Field(min_length=2, max_length=128)
    description: str = ""
    is_seeded: bool = False


class CollectionRead(BaseModel):
    """Public summary of a Collection workspace."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str
    is_seeded: bool
    document_count: int = 0
    created_at: datetime


class DocumentCreate(BaseModel):
    """Input payload for registering a newly uploaded or seeded document."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    collection_id: str
    filename: str
    doc_type: Literal["pdf", "md", "txt"]
    language: Literal["en", "ms", "mixed"] = "mixed"
    content_hash: str


class DocumentRead(BaseModel):
    """Persisted DocumentRecord metadata exposed by status and listing endpoints."""

    model_config = ConfigDict(from_attributes=True)

    doc_id: str
    collection_id: str
    filename: str
    doc_type: Literal["pdf", "md", "txt"]
    language: Literal["en", "ms", "mixed"]
    content_hash: str
    page_count: int = 0
    chunk_count: int = 0
    status: DocumentStatus
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class ParentChunkRecord(BaseModel):
    """Structure-preserving ParentChunk stored in SQLite for full LLM hydration."""

    model_config = ConfigDict(from_attributes=True)

    parent_id: str
    doc_id: str
    collection_id: str
    filename: str
    page_number: int
    section_header: str
    breadcrumb: str
    content: str
    token_count: int


class ChildChunkRecord(BaseModel):
    """Search-optimized ChildChunk prefixed with a Breadcrumb for Qdrant hybrid indexing."""

    model_config = ConfigDict(frozen=True)

    child_id: str
    parent_id: str
    doc_id: str
    collection_id: str
    filename: str
    page_number: int
    language: Literal["en", "ms", "mixed"]
    doc_type: Literal["pdf", "md", "txt"]
    breadcrumb: str
    content: str


class ChatSessionRead(BaseModel):
    """Persisted conversation thread scoped to one Collection."""

    model_config = ConfigDict(from_attributes=True)

    session_id: str
    collection_id: str
    title: str
    created_at: datetime


class ChatMessageRead(BaseModel):
    """Persisted user or assistant turn with citations and inference telemetry."""

    model_config = ConfigDict(from_attributes=True)

    message_id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    inference_mode: Literal["LOCAL_ONPREM", "CLOUD_API"]
    citations: list[dict[str, Any]] = Field(default_factory=list)
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    cache_hit_tokens: int = 0
    latency_ms: int = 0
    created_at: datetime
