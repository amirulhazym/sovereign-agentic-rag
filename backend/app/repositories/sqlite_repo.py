"""Async SQLite repository using SQLAlchemy 2.0 + aiosqlite.

Stores Collection workspaces, SHA-256 deduplicated DocumentRecords, full
ParentChunk payloads (preventing table/clause truncation in Qdrant), and chat history.
"""

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    func,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.exceptions import (
    CollectionConflictError,
    CollectionNotFoundError,
    DocumentNotFoundError,
)
from app.schemas.workspace import (
    ChatMessageRead,
    ChatSessionRead,
    CollectionCreate,
    CollectionRead,
    DocumentCreate,
    DocumentRead,
    DocumentStatus,
    ParentChunkRecord,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for async SQLite ORM models."""


class CollectionORM(Base):
    __tablename__ = "collections"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    is_seeded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class DocumentORM(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("collection_id", "content_hash", name="uq_collection_content_hash"),
    )

    doc_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    collection_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("collections.id", ondelete="CASCADE"), index=True, nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(16), nullable=False)
    language: Mapped[str] = mapped_column(String(16), default="mixed", nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(24), default=DocumentStatus.PENDING.value)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now
    )


class ParentChunkORM(Base):
    __tablename__ = "parent_chunks"

    parent_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True, nullable=False
    )
    collection_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    section_header: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    breadcrumb: Mapped[str] = mapped_column(String(512), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)


class ChatSessionORM(Base):
    __tablename__ = "chat_sessions"

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    collection_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("collections.id", ondelete="CASCADE"), index=True, nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), default="New Workspace Chat", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class ChatMessageORM(Base):
    __tablename__ = "chat_messages"

    message_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    inference_mode: Mapped[str] = mapped_column(String(24), nullable=False)
    citations_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    agent_trace_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    cache_hit_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class SQLiteRepository:
    """Async persistence adapter for workspace state, parent chunks, and chat history."""

    def __init__(self, sqlite_url: str) -> None:
        self._engine: AsyncEngine = create_async_engine(sqlite_url, future=True)

        @event.listens_for(self._engine.sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_conn: Any, _connection_record: Any) -> None:
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def init_schema(self) -> None:
        """Create tables if they do not exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def ping(self) -> bool:
        """Verify SQLite database responsiveness for /ready probes."""
        async with self._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True

    async def close(self) -> None:
        """Dispose connection pool during FastAPI lifespan shutdown."""
        await self._engine.dispose()

    async def create_collection(self, payload: CollectionCreate) -> CollectionRead:
        row = CollectionORM(
            id=payload.id,
            name=payload.name,
            description=payload.description,
            is_seeded=payload.is_seeded,
        )
        async with self._session_factory() as session:
            session.add(row)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise CollectionConflictError(payload.id) from exc
            await session.refresh(row)
        return CollectionRead(
            id=row.id,
            name=row.name,
            description=row.description,
            is_seeded=row.is_seeded,
            document_count=0,
            created_at=row.created_at,
        )

    async def get_collection(self, collection_id: str) -> CollectionRead:
        async with self._session_factory() as session:
            row = await session.get(CollectionORM, collection_id)
            if row is None:
                raise CollectionNotFoundError(collection_id)
            count_stmt = select(func.count(DocumentORM.doc_id)).where(
                DocumentORM.collection_id == collection_id
            )
            doc_count = (await session.execute(count_stmt)).scalar_one()
        return CollectionRead(
            id=row.id,
            name=row.name,
            description=row.description,
            is_seeded=row.is_seeded,
            document_count=doc_count,
            created_at=row.created_at,
        )

    async def list_collections(self) -> list[CollectionRead]:
        async with self._session_factory() as session:
            stmt = (
                select(CollectionORM, func.count(DocumentORM.doc_id))
                .outerjoin(DocumentORM, CollectionORM.id == DocumentORM.collection_id)
                .group_by(CollectionORM.id)
                .order_by(CollectionORM.created_at.asc())
            )
            results = (await session.execute(stmt)).all()
        return [
            CollectionRead(
                id=col.id,
                name=col.name,
                description=col.description,
                is_seeded=col.is_seeded,
                document_count=doc_count,
                created_at=col.created_at,
            )
            for col, doc_count in results
        ]

    async def create_document(self, payload: DocumentCreate) -> DocumentRead:
        await self.get_collection(payload.collection_id)
        row = DocumentORM(
            doc_id=payload.doc_id,
            collection_id=payload.collection_id,
            filename=payload.filename,
            doc_type=payload.doc_type,
            language=payload.language,
            content_hash=payload.content_hash,
            status=DocumentStatus.PENDING.value,
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
        return self._to_document_read(row)

    async def get_document(self, doc_id: str) -> DocumentRead:
        async with self._session_factory() as session:
            row = await session.get(DocumentORM, doc_id)
            if row is None:
                raise DocumentNotFoundError(doc_id)
        return self._to_document_read(row)

    async def get_document_by_hash(
        self, collection_id: str, content_hash: str
    ) -> DocumentRead | None:
        async with self._session_factory() as session:
            stmt = select(DocumentORM).where(
                DocumentORM.collection_id == collection_id,
                DocumentORM.content_hash == content_hash,
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
        return self._to_document_read(row) if row is not None else None

    async def list_collection_documents(self, collection_id: str) -> list[DocumentRead]:
        await self.get_collection(collection_id)
        async with self._session_factory() as session:
            stmt = (
                select(DocumentORM)
                .where(DocumentORM.collection_id == collection_id)
                .order_by(DocumentORM.created_at.desc())
            )
            rows = (await session.execute(stmt)).scalars().all()
        return [self._to_document_read(row) for row in rows]

    async def update_document_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        *,
        page_count: int | None = None,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> DocumentRead:
        async with self._session_factory() as session:
            row = await session.get(DocumentORM, doc_id)
            if row is None:
                raise DocumentNotFoundError(doc_id)
            row.status = status.value
            if page_count is not None:
                row.page_count = page_count
            if chunk_count is not None:
                row.chunk_count = chunk_count
            if error_message is not None:
                row.error_message = error_message
            row.updated_at = _utc_now()
            await session.commit()
            await session.refresh(row)
        return self._to_document_read(row)

    async def upsert_parent_chunks(self, parents: list[ParentChunkRecord]) -> None:
        if not parents:
            return
        async with self._session_factory() as session:
            for item in parents:
                await session.merge(
                    ParentChunkORM(
                        parent_id=item.parent_id,
                        doc_id=item.doc_id,
                        collection_id=item.collection_id,
                        filename=item.filename,
                        page_number=item.page_number,
                        section_header=item.section_header,
                        breadcrumb=item.breadcrumb,
                        content=item.content,
                        token_count=item.token_count,
                    )
                )
            await session.commit()

    async def get_parent_chunks_by_ids(self, parent_ids: list[str]) -> list[ParentChunkRecord]:
        """Hydrate unique ParentChunks deterministically sorted by (doc_id, page_number, parent_id).

        Deterministic ordering preserves DeepSeek Prefix Context Caching across turns.
        """
        unique_ids = list(dict.fromkeys(parent_ids))
        if not unique_ids:
            return []
        async with self._session_factory() as session:
            stmt = select(ParentChunkORM).where(ParentChunkORM.parent_id.in_(unique_ids))
            rows = (await session.execute(stmt)).scalars().all()
        records = [ParentChunkRecord.model_validate(row) for row in rows]
        records.sort(key=lambda r: (r.doc_id, r.page_number, r.parent_id))
        return records

    async def ensure_chat_session(
        self, session_id: str, collection_id: str, title: str = "New Workspace Chat"
    ) -> ChatSessionRead:
        await self.get_collection(collection_id)
        async with self._session_factory() as session:
            row = await session.get(ChatSessionORM, session_id)
            if row is None:
                row = ChatSessionORM(
                    session_id=session_id,
                    collection_id=collection_id,
                    title=title,
                )
                session.add(row)
                await session.commit()
                await session.refresh(row)
        return ChatSessionRead.model_validate(row)

    async def append_chat_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        inference_mode: str,
        citations: list[dict[str, Any]] | None = None,
        agent_trace: list[dict[str, Any]] | None = None,
        cache_hit_tokens: int = 0,
        latency_ms: int = 0,
        message_id: str | None = None,
    ) -> ChatMessageRead:
        resolved_id = message_id or f"msg_{uuid4().hex[:12]}"
        row = ChatMessageORM(
            message_id=resolved_id,
            session_id=session_id,
            role=role,
            content=content,
            inference_mode=inference_mode,
            citations_json=json.dumps(citations or []),
            agent_trace_json=json.dumps(agent_trace or []),
            cache_hit_tokens=cache_hit_tokens,
            latency_ms=latency_ms,
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
        return self._to_chat_message_read(row)

    async def list_session_messages(self, session_id: str) -> list[ChatMessageRead]:
        async with self._session_factory() as session:
            stmt = (
                select(ChatMessageORM)
                .where(ChatMessageORM.session_id == session_id)
                .order_by(ChatMessageORM.created_at.asc())
            )
            rows = (await session.execute(stmt)).scalars().all()
        return [self._to_chat_message_read(row) for row in rows]

    @staticmethod
    def _to_document_read(row: DocumentORM) -> DocumentRead:
        return DocumentRead(
            doc_id=row.doc_id,
            collection_id=row.collection_id,
            filename=row.filename,
            doc_type=row.doc_type,  # type: ignore[arg-type]
            language=row.language,  # type: ignore[arg-type]
            content_hash=row.content_hash,
            page_count=row.page_count,
            chunk_count=row.chunk_count,
            status=DocumentStatus(row.status),
            error_message=row.error_message,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _to_chat_message_read(row: ChatMessageORM) -> ChatMessageRead:
        return ChatMessageRead(
            message_id=row.message_id,
            session_id=row.session_id,
            role=row.role,  # type: ignore[arg-type]
            content=row.content,
            inference_mode=row.inference_mode,  # type: ignore[arg-type]
            citations=json.loads(row.citations_json),
            agent_trace=json.loads(row.agent_trace_json),
            cache_hit_tokens=row.cache_hit_tokens,
            latency_ms=row.latency_ms,
            created_at=row.created_at,
        )
