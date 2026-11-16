"""VectorRepository seam implementing Qdrant Hybrid RRF and an in-memory RRF fake.

Isolates all Qdrant Prefetch/FusionQuery details and mandatory `collection_id`
tenant filtering behind a narrow interface so LangGraph and tests stay decoupled.
"""

from math import sqrt
from typing import Protocol
from uuid import NAMESPACE_DNS, uuid5

from pydantic import BaseModel, ConfigDict
from qdrant_client import AsyncQdrantClient, models

from app.schemas.workspace import ChildChunkRecord

_RRF_K = 60


class SparseVectorData(BaseModel):
    """Sparse lexical vector representation compatible with Qdrant SparseVector."""

    model_config = ConfigDict(frozen=True)

    indices: list[int]
    values: list[float]


class VectorChunkUpsert(BaseModel):
    """Bundled ChildChunk payload with precomputed dense and sparse vectors."""

    model_config = ConfigDict(frozen=True)

    chunk: ChildChunkRecord
    dense_vector: list[float]
    sparse_vector: SparseVectorData


class HybridSearchHit(BaseModel):
    """Single ChildChunk hit returned from hybrid RRF search."""

    model_config = ConfigDict(frozen=True)

    child_id: str
    parent_id: str
    doc_id: str
    collection_id: str
    filename: str
    page_number: int
    language: str
    doc_type: str
    breadcrumb: str
    content: str
    score: float


class VectorRepository(Protocol):
    """Seam for tenant-isolated hybrid dense+sparse retrieval."""

    async def ensure_collection(self) -> None: ...

    async def ping(self) -> bool: ...

    async def upsert_chunks(self, items: list[VectorChunkUpsert]) -> None: ...

    async def hybrid_search(
        self,
        *,
        collection_id: str,
        dense_vector: list[float],
        sparse_vector: SparseVectorData,
        top_k: int = 12,
        language: str | None = None,
        doc_type: str | None = None,
    ) -> list[HybridSearchHit]: ...

    async def delete_document(self, collection_id: str, doc_id: str) -> None: ...


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _sparse_dot_product(a: SparseVectorData, b: SparseVectorData) -> float:
    if not a.indices or not b.indices:
        return 0.0
    b_map = dict(zip(b.indices, b.values, strict=True))
    return sum(val * b_map.get(idx, 0.0) for idx, val in zip(a.indices, a.values, strict=True))


class InMemoryVectorRepository:
    """Deterministic in-memory VectorRepository performing exact tenant filtering and RRF."""

    def __init__(self) -> None:
        self._items: dict[str, VectorChunkUpsert] = {}

    async def ensure_collection(self) -> None:
        return None

    async def ping(self) -> bool:
        return True

    async def upsert_chunks(self, items: list[VectorChunkUpsert]) -> None:
        for item in items:
            self._items[item.chunk.child_id] = item

    async def delete_document(self, collection_id: str, doc_id: str) -> None:
        to_delete = [
            cid
            for cid, item in self._items.items()
            if item.chunk.collection_id == collection_id and item.chunk.doc_id == doc_id
        ]
        for cid in to_delete:
            del self._items[cid]

    async def hybrid_search(
        self,
        *,
        collection_id: str,
        dense_vector: list[float],
        sparse_vector: SparseVectorData,
        top_k: int = 12,
        language: str | None = None,
        doc_type: str | None = None,
    ) -> list[HybridSearchHit]:
        filtered = [
            item
            for item in self._items.values()
            if item.chunk.collection_id == collection_id
            and (language is None or item.chunk.language == language)
            and (doc_type is None or item.chunk.doc_type == doc_type)
        ]
        if not filtered:
            return []

        dense_ranked = sorted(
            filtered,
            key=lambda it: _cosine_similarity(dense_vector, it.dense_vector),
            reverse=True,
        )[:25]
        sparse_candidates = [
            (it, _sparse_dot_product(sparse_vector, it.sparse_vector)) for it in filtered
        ]
        sparse_ranked = [
            it
            for it, score in sorted(sparse_candidates, key=lambda pair: pair[1], reverse=True)[:25]
            if score > 0.0
        ]

        rrf_scores: dict[str, float] = {}
        item_lookup: dict[str, VectorChunkUpsert] = {}

        for rank, item in enumerate(dense_ranked, start=1):
            cid = item.chunk.child_id
            item_lookup[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)

        for rank, item in enumerate(sparse_ranked, start=1):
            cid = item.chunk.child_id
            item_lookup[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)

        ordered = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
        return [
            HybridSearchHit(
                child_id=item_lookup[cid].chunk.child_id,
                parent_id=item_lookup[cid].chunk.parent_id,
                doc_id=item_lookup[cid].chunk.doc_id,
                collection_id=item_lookup[cid].chunk.collection_id,
                filename=item_lookup[cid].chunk.filename,
                page_number=item_lookup[cid].chunk.page_number,
                language=item_lookup[cid].chunk.language,
                doc_type=item_lookup[cid].chunk.doc_type,
                breadcrumb=item_lookup[cid].chunk.breadcrumb,
                content=item_lookup[cid].chunk.content,
                score=score,
            )
            for cid, score in ordered
        ]


class QdrantHybridRepository:
    """Production Qdrant repository using named Dense + Sparse BM25 vectors fused via RRF."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str = "sovereign_rag_chunks",
        dense_dim: int = 1024,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._dense_dim = dense_dim

    async def ping(self) -> bool:
        await self._client.get_collections()
        return True

    async def ensure_collection(self) -> None:
        exists = await self._client.collection_exists(self._collection_name)
        if exists:
            return
        await self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self._dense_dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        keyword_fields = ["collection_id", "doc_id", "parent_id", "language", "doc_type"]
        for field in keyword_fields:
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        await self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="page_number",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

    async def upsert_chunks(self, items: list[VectorChunkUpsert]) -> None:
        if not items:
            return
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_DNS, item.chunk.child_id)),
                vector={
                    "dense": item.dense_vector,
                    "sparse": models.SparseVector(
                        indices=item.sparse_vector.indices,
                        values=item.sparse_vector.values,
                    ),
                },
                payload=item.chunk.model_dump(),
            )
            for item in items
        ]
        await self._client.upsert(collection_name=self._collection_name, points=points)

    async def delete_document(self, collection_id: str, doc_id: str) -> None:
        doc_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="collection_id", match=models.MatchValue(value=collection_id)
                ),
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            ]
        )
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(filter=doc_filter),
        )

    async def hybrid_search(
        self,
        *,
        collection_id: str,
        dense_vector: list[float],
        sparse_vector: SparseVectorData,
        top_k: int = 12,
        language: str | None = None,
        doc_type: str | None = None,
    ) -> list[HybridSearchHit]:
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key="collection_id", match=models.MatchValue(value=collection_id)
            )
        ]
        if language is not None:
            conditions.append(
                models.FieldCondition(key="language", match=models.MatchValue(value=language))
            )
        if doc_type is not None:
            conditions.append(
                models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type))
            )
        query_filter = models.Filter(must=conditions)

        prefetch_limit = max(top_k * 2, 25)
        response = await self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values,
                    ),
                    using="sparse",
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        hits: list[HybridSearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                HybridSearchHit(
                    child_id=str(payload.get("child_id", point.id)),
                    parent_id=str(payload["parent_id"]),
                    doc_id=str(payload["doc_id"]),
                    collection_id=str(payload["collection_id"]),
                    filename=str(payload.get("filename", "")),
                    page_number=int(payload.get("page_number", 1)),
                    language=str(payload.get("language", "mixed")),
                    doc_type=str(payload.get("doc_type", "pdf")),
                    breadcrumb=str(payload.get("breadcrumb", "")),
                    content=str(payload.get("content", "")),
                    score=float(point.score),
                )
            )
        return hits
