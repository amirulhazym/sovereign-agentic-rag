"""Tests for async SQLiteRepository (collections, SHA-256 docs, parent chunks, chat history)."""

from pathlib import Path

import pytest

from app.core.exceptions import (
    CollectionConflictError,
    CollectionNotFoundError,
    DocumentNotFoundError,
)
from app.repositories.sqlite_repo import SQLiteRepository
from app.schemas.workspace import (
    CollectionCreate,
    DocumentCreate,
    DocumentStatus,
    ParentChunkRecord,
)


@pytest.fixture
async def sqlite_repo(tmp_path: Path) -> SQLiteRepository:
    """Create a fresh SQLiteRepository backed by an isolated temporary database."""
    db_file = tmp_path / "test_rag.db"
    repo = SQLiteRepository(f"sqlite+aiosqlite:///{db_file.as_posix()}")
    await repo.init_schema()
    try:
        yield repo
    finally:
        await repo.close()


async def test_collection_crud_and_conflict_handling(sqlite_repo: SQLiteRepository) -> None:
    """Verify creating, listing, fetching, and rejecting duplicate collection IDs."""
    created = await sqlite_repo.create_collection(
        CollectionCreate(
            id="it_infra_sops",
            name="Enterprise IT & Server SOPs",
            description="Hardware specs and deployment runbooks",
            is_seeded=True,
        )
    )
    assert created.id == "it_infra_sops"
    assert created.is_seeded is True

    with pytest.raises(CollectionConflictError):
        await sqlite_repo.create_collection(
            CollectionCreate(id="it_infra_sops", name="Duplicate", description="dup")
        )

    with pytest.raises(CollectionNotFoundError):
        await sqlite_repo.get_collection("nonexistent")

    all_collections = await sqlite_repo.list_collections()
    assert len(all_collections) == 1
    assert all_collections[0].document_count == 0


async def test_document_sha256_deduplication_and_lifecycle_transitions(
    sqlite_repo: SQLiteRepository,
) -> None:
    """Verify (collection_id, content_hash) uniqueness and status transitions."""
    await sqlite_repo.create_collection(
        CollectionCreate(id="col_a", name="Collection A", description="A")
    )
    await sqlite_repo.create_collection(
        CollectionCreate(id="col_b", name="Collection B", description="B")
    )

    doc_a = await sqlite_repo.create_document(
        DocumentCreate(
            doc_id="doc_sop_1",
            collection_id="col_a",
            filename="nvme_sop.pdf",
            doc_type="pdf",
            language="en",
            content_hash="abc123sha256",
        )
    )
    assert doc_a.status == DocumentStatus.PENDING

    # Same hash inside the same collection is discoverable for idempotent return
    found_dup = await sqlite_repo.get_document_by_hash("col_a", "abc123sha256")
    assert found_dup is not None
    assert found_dup.doc_id == "doc_sop_1"

    # Same hash in a DIFFERENT collection must be allowed (strict tenant isolation)
    doc_b = await sqlite_repo.create_document(
        DocumentCreate(
            doc_id="doc_sop_2",
            collection_id="col_b",
            filename="nvme_sop.pdf",
            doc_type="pdf",
            language="en",
            content_hash="abc123sha256",
        )
    )
    assert doc_b.collection_id == "col_b"

    await sqlite_repo.update_document_status("doc_sop_1", DocumentStatus.INDEXING)
    indexing_doc = await sqlite_repo.get_document("doc_sop_1")
    assert indexing_doc.status == DocumentStatus.INDEXING

    await sqlite_repo.update_document_status(
        "doc_sop_1",
        DocumentStatus.READY,
        page_count=12,
        chunk_count=34,
    )
    ready_doc = await sqlite_repo.get_document("doc_sop_1")
    assert ready_doc.status == DocumentStatus.READY
    assert ready_doc.page_count == 12
    assert ready_doc.chunk_count == 34

    await sqlite_repo.update_document_status(
        "doc_sop_2",
        DocumentStatus.FAILED,
        error_message="Corrupt PDF xref table",
    )
    failed_doc = await sqlite_repo.get_document("doc_sop_2")
    assert failed_doc.status == DocumentStatus.FAILED
    assert failed_doc.error_message == "Corrupt PDF xref table"

    with pytest.raises(DocumentNotFoundError):
        await sqlite_repo.get_document("missing_doc")


async def test_parent_chunk_storage_and_deterministic_hydration(
    sqlite_repo: SQLiteRepository,
) -> None:
    """Verify ParentChunks hydrate sorted deterministically by (doc_id, page_number, parent_id)."""
    await sqlite_repo.create_collection(
        CollectionCreate(id="policy_ws", name="Malaysian Policy", description="MY")
    )
    await sqlite_repo.create_document(
        DocumentCreate(
            doc_id="hr_my",
            collection_id="policy_ws",
            filename="hr_policy.pdf",
            doc_type="pdf",
            language="mixed",
            content_hash="hash_hr",
        )
    )

    parents = [
        ParentChunkRecord(
            parent_id="hr_my_p12_s0",
            doc_id="hr_my",
            collection_id="policy_ws",
            filename="hr_policy.pdf",
            page_number=12,
            section_header="Overtime Claims",
            breadcrumb="hr_policy.pdf | Page 12 > Overtime Claims",
            content="Klausa 4.2: Bayaran 2x untuk kerja pada hari kelepasan am.",
            token_count=18,
        ),
        ParentChunkRecord(
            parent_id="hr_my_p3_s0",
            doc_id="hr_my",
            collection_id="policy_ws",
            filename="hr_policy.pdf",
            page_number=3,
            section_header="Working Hours",
            breadcrumb="hr_policy.pdf | Page 3 > Working Hours",
            content="Waktu kerja biasa ialah 9 pagi hingga 6 petang.",
            token_count=12,
        ),
    ]
    await sqlite_repo.upsert_parent_chunks(parents)

    hydrated = await sqlite_repo.get_parent_chunks_by_ids(
        ["hr_my_p12_s0", "hr_my_p3_s0", "hr_my_p12_s0"]
    )
    # Deduplicated and ordered deterministically (page 3 before page 12) for DeepSeek caching
    assert [p.parent_id for p in hydrated] == ["hr_my_p3_s0", "hr_my_p12_s0"]
    assert hydrated[1].page_number == 12


async def test_chat_session_and_message_persistence(sqlite_repo: SQLiteRepository) -> None:
    """Verify chat turns record citations, inference mode, cache tokens, and latency."""
    await sqlite_repo.create_collection(
        CollectionCreate(id="policy_ws", name="Malaysian Policy", description="MY")
    )
    session = await sqlite_repo.ensure_chat_session("sess_01", "policy_ws", "OT Inquiry")
    assert session.session_id == "sess_01"

    await sqlite_repo.append_chat_message(
        session_id="sess_01",
        role="user",
        content="Apa prosedur tuntutan OT?",
        inference_mode="CLOUD_API",
    )
    await sqlite_repo.append_chat_message(
        session_id="sess_01",
        role="assistant",
        content="Pekerja layak menuntut bayaran 2x [hr_my:p12].",
        inference_mode="CLOUD_API",
        citations=[{"citation_key": "[hr_my:p12]", "doc_id": "hr_my", "page": 12}],
        agent_trace=[{"node": "QueryRouter", "latency_ms": 120}],
        cache_hit_tokens=1280,
        latency_ms=1450,
    )

    history = await sqlite_repo.list_session_messages("sess_01")
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].cache_hit_tokens == 1280
    assert history[1].citations[0]["citation_key"] == "[hr_my:p12]"
