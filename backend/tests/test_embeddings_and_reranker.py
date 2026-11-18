"""Tests for lazy FastEmbed ONNX EmbeddingProvider, ONNXRerankerProvider, and fakes."""

from types import SimpleNamespace

from app.providers.embeddings import (
    FakeEmbeddingProvider,
    FakeRerankerProvider,
    FastEmbedProvider,
    ONNXRerankerProvider,
    _ensure_bge_m3_registered,
    _resolve_reranker_model_name,
)
from app.schemas.workspace import ParentChunkRecord


def _parent(pid: str, content: str, page: int = 1) -> ParentChunkRecord:
    return ParentChunkRecord(
        parent_id=pid,
        doc_id="doc_1",
        collection_id="col_a",
        filename="doc_1.pdf",
        page_number=page,
        section_header="Section",
        breadcrumb=f"doc_1.pdf | Page {page} > Section",
        content=content,
        token_count=len(content.split()),
    )


async def test_fastembed_provider_lazy_loads_models_only_on_first_call() -> None:
    """Verify zero model instantiation at init time and proper dense/sparse formatting."""
    load_counts = {"dense": 0, "sparse": 0}

    class _MockDense:
        def embed(self, texts: list[str]):
            return [[0.5, 0.5] for _ in texts]

    class _MockSparse:
        def embed(self, texts: list[str]):
            return [SimpleNamespace(indices=[1, 7], values=[1.2, 0.4]) for _ in texts]

    def _dense_loader():
        load_counts["dense"] += 1
        return _MockDense()

    def _sparse_loader():
        load_counts["sparse"] += 1
        return _MockSparse()

    provider = FastEmbedProvider(
        dense_loader=_dense_loader,
        sparse_loader=_sparse_loader,
    )
    assert load_counts == {"dense": 0, "sparse": 0}

    dense_vecs = await provider.embed_dense(["test query", "second text"])
    assert dense_vecs == [[0.5, 0.5], [0.5, 0.5]]
    assert load_counts == {"dense": 1, "sparse": 0}

    sparse_vecs = await provider.embed_sparse(["test query"])
    assert len(sparse_vecs) == 1
    assert sparse_vecs[0].indices == [1, 7]
    assert sparse_vecs[0].values == [1.2, 0.4]
    assert load_counts == {"dense": 1, "sparse": 1}

    # Cached session reused on subsequent calls
    await provider.embed_dense(["again"])
    assert load_counts["dense"] == 1


async def test_onnx_reranker_lazy_loads_and_sorts_parents_descending() -> None:
    """Verify ONNXRerankerProvider defers loading and returns top_k sorted ScoredParents."""
    loaded = {"count": 0}

    class _MockCrossEncoder:
        def rerank(self, query: str, documents: list[str]):
            return [0.15, 0.92, 0.48][: len(documents)]

    def _loader():
        loaded["count"] += 1
        return _MockCrossEncoder()

    reranker = ONNXRerankerProvider(model_loader=_loader)
    assert loaded["count"] == 0

    candidates = [
        _parent("p1", "Irrelevant preamble", page=1),
        _parent("p2", "Klausa 4.2: Overtime public holiday 2x", page=12),
        _parent("p3", "General working hours", page=3),
    ]
    ranked = await reranker.rerank("tuntutan OT public holiday", candidates, top_k=2)

    assert loaded["count"] == 1
    assert [item.parent.parent_id for item in ranked] == ["p2", "p3"]
    assert ranked[0].score == 0.92
    assert ranked[1].score == 0.48


async def test_fake_embedding_and_reranker_are_deterministic() -> None:
    """Verify deterministic test doubles work without ONNX weights."""
    embedder = FakeEmbeddingProvider(dense_dim=8)
    d1 = await embedder.embed_dense(["NVMe error code"])
    d2 = await embedder.embed_dense(["NVMe error code"])
    assert len(d1[0]) == 8
    assert d1 == d2

    s1 = await embedder.embed_sparse(["NVMe ERR_NVME_0x04"])
    assert len(s1[0].indices) >= 2

    reranker = FakeRerankerProvider()
    candidates = [
        _parent("p1", "Cafeteria menu", page=1),
        _parent("p2", "NVMe ERR_NVME_0x04 hot-swap runbook", page=4),
    ]
    ranked = await reranker.rerank("NVMe ERR_NVME_0x04", candidates, top_k=1)
    assert ranked[0].parent.parent_id == "p2"


def test_ensure_bge_m3_registered_adds_custom_onnx_model_to_fastembed() -> None:
    """Verify BAAI/bge-m3 quantized Xenova ONNX is registered idempotently and int8 resolves."""
    from fastembed import TextEmbedding

    _ensure_bge_m3_registered()
    _ensure_bge_m3_registered()  # Idempotent repeat call must not raise

    matched = [m for m in TextEmbedding.list_supported_models() if m["model"] == "BAAI/bge-m3"]
    assert len(matched) == 1
    assert matched[0]["dim"] == 1024
    assert matched[0]["sources"]["hf"] == "Xenova/bge-m3"

    assert _resolve_reranker_model_name("BAAI/bge-reranker-v2-m3") == "BAAI/bge-reranker-v2-m3-int8"
    assert _resolve_reranker_model_name("BAAI/bge-reranker-base") == "BAAI/bge-reranker-base"
