"""Non-blocking document ingestion service with SHA-256 deduplication and status tracking."""

import asyncio
import hashlib
import logging
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from app.providers.embeddings import EmbeddingProvider
from app.repositories.qdrant_repo import VectorChunkUpsert, VectorRepository
from app.repositories.sqlite_repo import SQLiteRepository
from app.schemas.workspace import DocumentCreate, DocumentRead, DocumentStatus
from app.services.chunking import (
    HierarchicalChunker,
    detect_doc_type,
    detect_language,
    extract_pages,
)

logger = logging.getLogger(__name__)


class UploadDispatchResult(BaseModel):
    """Response returned immediately when a document upload is queued or deduplicated."""

    model_config = ConfigDict(frozen=True)

    document: DocumentRead
    deduplicated: bool


class DocumentIngestionService:
    """Deep module orchestrating deduplication, PyMuPDF4LLM parsing, chunking, and indexing."""

    def __init__(
        self,
        *,
        sqlite_repo: SQLiteRepository,
        vector_repo: VectorRepository,
        embedding_provider: EmbeddingProvider,
        chunker: HierarchicalChunker | None = None,
    ) -> None:
        self._sqlite = sqlite_repo
        self._vectors = vector_repo
        self._embedder = embedding_provider
        self._chunker = chunker or HierarchicalChunker()

    async def prepare_upload(
        self,
        *,
        collection_id: str,
        filename: str,
        file_bytes: bytes,
        doc_id: str | None = None,
    ) -> UploadDispatchResult:
        """Validate collection/file type and register or deduplicate DocumentRecord."""
        await self._sqlite.get_collection(collection_id)
        doc_type = detect_doc_type(filename)
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        existing = await self._sqlite.get_document_by_hash(collection_id, content_hash)
        if existing is not None:
            if existing.status == DocumentStatus.FAILED:
                reset_doc = await self._sqlite.update_document_status(
                    existing.doc_id,
                    DocumentStatus.PENDING,
                    error_message="",
                )
                return UploadDispatchResult(document=reset_doc, deduplicated=False)
            return UploadDispatchResult(document=existing, deduplicated=True)

        resolved_doc_id = doc_id or f"doc_{uuid4().hex[:12]}"
        created = await self._sqlite.create_document(
            DocumentCreate(
                doc_id=resolved_doc_id,
                collection_id=collection_id,
                filename=filename,
                doc_type=doc_type,
                language="mixed",
                content_hash=content_hash,
            )
        )
        return UploadDispatchResult(document=created, deduplicated=False)

    async def process_document(
        self,
        *,
        doc_id: str,
        collection_id: str,
        filename: str,
        file_bytes: bytes,
    ) -> DocumentRead:
        """Run CPU parsing/chunking off the event loop, embed, and persist to SQLite + Qdrant."""
        await self._sqlite.update_document_status(doc_id, DocumentStatus.INDEXING)
        try:
            doc_type = detect_doc_type(filename)
            pages, parents, children, detected_lang = await asyncio.to_thread(
                self._extract_and_chunk_sync,
                file_bytes,
                filename,
                doc_id,
                collection_id,
                doc_type,
            )

            await self._vectors.ensure_collection()
            await self._sqlite.upsert_parent_chunks(parents)

            child_texts = [c.content for c in children]
            dense_vecs, sparse_vecs = await asyncio.gather(
                self._embedder.embed_dense(child_texts),
                self._embedder.embed_sparse(child_texts),
            )
            upserts = [
                VectorChunkUpsert(
                    chunk=child,
                    dense_vector=dense_vec,
                    sparse_vector=sparse_vec,
                )
                for child, dense_vec, sparse_vec in zip(
                    children, dense_vecs, sparse_vecs, strict=True
                )
            ]
            await self._vectors.upsert_chunks(upserts)

            return await self._sqlite.update_document_status(
                doc_id,
                DocumentStatus.READY,
                language=detected_lang,
                page_count=len(pages),
                chunk_count=len(children),
            )
        except Exception as exc:
            logger.exception("Ingestion failed for doc_id=%s (%s)", doc_id, filename)
            return await self._sqlite.update_document_status(
                doc_id,
                DocumentStatus.FAILED,
                error_message=str(exc),
            )

    def _extract_and_chunk_sync(
        self,
        file_bytes: bytes,
        filename: str,
        doc_id: str,
        collection_id: str,
        doc_type: str,
    ):
        pages = extract_pages(file_bytes, filename)
        if not pages:
            raise ValueError(f"No readable text extracted from '{filename}'.")
        combined_text = "\n".join(page.markdown for page in pages)
        language = detect_language(combined_text)
        parents, children = self._chunker.chunk_document(
            pages=pages,
            doc_id=doc_id,
            collection_id=collection_id,
            filename=filename,
            doc_type=doc_type,  # type: ignore[arg-type]
            language=language,
        )
        return pages, parents, children, language
