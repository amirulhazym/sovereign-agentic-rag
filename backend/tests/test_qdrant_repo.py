"""Tests for VectorRepository seam (InMemoryVectorRepository RRF & QdrantHybridRepository)."""

from unittest.mock import AsyncMock, MagicMock

from qdrant_client import models

from app.repositories.qdrant_repo import (
    InMemoryVectorRepository,
    QdrantHybridRepository,
    SparseVectorData,
    VectorChunkUpsert,
)
from app.schemas.workspace import ChildChunkRecord


def _make_child(
    child_id: str,
    parent_id: str,
    doc_id: str,
    collection_id: str,
    *,
    language: str = "en",
    doc_type: str = "md",
    page_number: int = 1,
) -> ChildChunkRecord:
    return ChildChunkRecord(
        child_id=child_id,
        parent_id=parent_id,
        doc_id=doc_id,
        collection_id=collection_id,
        filename=f"{doc_id}.{doc_type}",
        page_number=page_number,
        language=language,  # type: ignore[arg-type]
        doc_type=doc_type,  # type: ignore[arg-type]
        breadcrumb=f"{doc_id} > Page {page_number}",
        content=f"Content for {child_id}",
    )


async def test_inmemory_repo_enforces_collection_isolation_and_rrf_ranking() -> None:
    """Verify collection_id tenant boundary and RRF combining dense + sparse matches."""
    repo = InMemoryVectorRepository()
    await repo.upsert_chunks(
        [
            VectorChunkUpsert(
                chunk=_make_child("c_dense_only", "p_dense", "doc_a", "col_a"),
                dense_vector=[1.0, 0.0, 0.0],
                sparse_vector=SparseVectorData(indices=[99], values=[0.1]),
            ),
            VectorChunkUpsert(
                chunk=_make_child("c_hybrid_hit", "p_hybrid", "doc_a", "col_a"),
                dense_vector=[0.9, 0.4, 0.0],
                sparse_vector=SparseVectorData(indices=[10, 20], values=[3.0, 2.0]),
            ),
            VectorChunkUpsert(
                chunk=_make_child("c_other_tenant", "p_other", "doc_b", "col_b"),
                dense_vector=[1.0, 0.0, 0.0],
                sparse_vector=SparseVectorData(indices=[10, 20], values=[5.0, 5.0]),
            ),
        ]
    )

    hits = await repo.hybrid_search(
        collection_id="col_a",
        dense_vector=[0.95, 0.3, 0.0],
        sparse_vector=SparseVectorData(indices=[10, 20], values=[1.0, 1.0]),
        top_k=5,
    )

    assert all(hit.collection_id == "col_a" for hit in hits)
    assert "c_other_tenant" not in [hit.child_id for hit in hits]
    # Hybrid hit appears high in BOTH dense & sparse lists -> wins RRF fusion
    assert hits[0].child_id == "c_hybrid_hit"
    assert hits[0].parent_id == "p_hybrid"


async def test_inmemory_repo_filters_and_deletes_by_doc_id() -> None:
    """Verify optional language/doc_type filters and doc_id deletion."""
    repo = InMemoryVectorRepository()
    await repo.upsert_chunks(
        [
            VectorChunkUpsert(
                chunk=_make_child("c1", "p1", "doc_1", "col_a", language="ms", doc_type="pdf"),
                dense_vector=[1.0, 0.0],
                sparse_vector=SparseVectorData(indices=[1], values=[1.0]),
            ),
            VectorChunkUpsert(
                chunk=_make_child("c2", "p2", "doc_2", "col_a", language="en", doc_type="md"),
                dense_vector=[1.0, 0.0],
                sparse_vector=SparseVectorData(indices=[1], values=[1.0]),
            ),
        ]
    )

    ms_hits = await repo.hybrid_search(
        collection_id="col_a",
        dense_vector=[1.0, 0.0],
        sparse_vector=SparseVectorData(indices=[1], values=[1.0]),
        language="ms",
        top_k=5,
    )
    assert [h.child_id for h in ms_hits] == ["c1"]

    await repo.delete_document("col_a", "doc_1")
    remaining = await repo.hybrid_search(
        collection_id="col_a",
        dense_vector=[1.0, 0.0],
        sparse_vector=SparseVectorData(indices=[1], values=[1.0]),
        top_k=5,
    )
    assert [h.child_id for h in remaining] == ["c2"]


async def test_qdrant_hybrid_repo_builds_rrf_prefetch_and_tenant_filter() -> None:
    """Verify QdrantHybridRepository configures named dense+sparse vectors and RRF query."""
    mock_client = AsyncMock()
    mock_client.collection_exists.return_value = False

    scored_point = MagicMock()
    scored_point.id = "c1"
    scored_point.score = 0.0325
    scored_point.payload = {
        "child_id": "c1",
        "parent_id": "p1",
        "doc_id": "doc_1",
        "collection_id": "it_infra_sops",
        "filename": "sop.pdf",
        "page_number": 4,
        "language": "en",
        "doc_type": "pdf",
        "breadcrumb": "sop.pdf | Page 4 > NVMe",
        "content": "NVMe error ERR_NVME_0x04 procedure",
    }
    query_response = MagicMock()
    query_response.points = [scored_point]
    mock_client.query_points.return_value = query_response

    repo = QdrantHybridRepository(
        client=mock_client,
        collection_name="sovereign_rag_chunks",
        dense_dim=1024,
    )
    await repo.ensure_collection()

    mock_client.create_collection.assert_awaited_once()
    create_kwargs = mock_client.create_collection.call_args.kwargs
    assert "dense" in create_kwargs["vectors_config"]
    assert "sparse" in create_kwargs["sparse_vectors_config"]

    results = await repo.hybrid_search(
        collection_id="it_infra_sops",
        dense_vector=[0.1, 0.2],
        sparse_vector=SparseVectorData(indices=[5, 8], values=[1.2, 0.8]),
        top_k=4,
    )

    assert len(results) == 1
    assert results[0].parent_id == "p1"
    assert results[0].score == 0.0325

    query_kwargs = mock_client.query_points.call_args.kwargs
    assert isinstance(query_kwargs["query"], models.FusionQuery)
    assert query_kwargs["query"].fusion == models.Fusion.RRF
    assert len(query_kwargs["prefetch"]) == 2
    assert {p.using for p in query_kwargs["prefetch"]} == {"dense", "sparse"}
    # Both prefetch branches must enforce collection_id filter before RRF fusion
    for prefetch in query_kwargs["prefetch"]:
        assert prefetch.filter is not None
        assert prefetch.filter.must[0].key == "collection_id"
