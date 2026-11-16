"""Lazy ONNX embedding and cross-encoder reranker providers via FastEmbed.

Using ONNX Runtime (FastEmbed) avoids loading multi-GB PyTorch weights in `backend/`,
protecting the 16GB RAM budget alongside `empero-ai/Qwen3.8-4B/9B-Distill-GGUF`.
"""

import asyncio
import hashlib
import re
from collections.abc import Callable
from math import sqrt
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from app.repositories.qdrant_repo import SparseVectorData
from app.schemas.workspace import ParentChunkRecord


class ScoredParent(BaseModel):
    """ParentChunk paired with its Cross-Encoder relevance score."""

    model_config = ConfigDict(frozen=True)

    parent: ParentChunkRecord
    score: float


class EmbeddingProvider(Protocol):
    """Seam producing dense semantic and sparse lexical vectors."""

    async def embed_dense(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_sparse(self, texts: list[str]) -> list[SparseVectorData]: ...


class RerankerProvider(Protocol):
    """Seam re-scoring candidate ParentChunks against the active query."""

    async def rerank(
        self,
        query: str,
        parents: list[ParentChunkRecord],
        top_k: int = 5,
    ) -> list[ScoredParent]: ...


def _to_float_list(vec: Any) -> list[float]:
    if hasattr(vec, "tolist"):
        return [float(x) for x in vec.tolist()]
    return [float(x) for x in vec]


def _to_int_list(vec: Any) -> list[int]:
    if hasattr(vec, "tolist"):
        return [int(x) for x in vec.tolist()]
    return [int(x) for x in vec]


class FastEmbedProvider:
    """Lazy ONNX dense + sparse embedding adapter running inference off the event loop."""

    def __init__(
        self,
        *,
        dense_model_name: str = "BAAI/bge-m3",
        sparse_model_name: str = "Qdrant/bm25",
        dense_loader: Callable[[], Any] | None = None,
        sparse_loader: Callable[[], Any] | None = None,
    ) -> None:
        self._dense_model_name = dense_model_name
        self._sparse_model_name = sparse_model_name
        self._dense_loader = dense_loader or self._default_dense_loader
        self._sparse_loader = sparse_loader or self._default_sparse_loader
        self._dense_model: Any | None = None
        self._sparse_model: Any | None = None

    def _default_dense_loader(self) -> Any:
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=self._dense_model_name)

    def _default_sparse_loader(self) -> Any:
        from fastembed import SparseTextEmbedding

        return SparseTextEmbedding(model_name=self._sparse_model_name)

    def _embed_dense_sync(self, texts: list[str]) -> list[list[float]]:
        if self._dense_model is None:
            self._dense_model = self._dense_loader()
        return [_to_float_list(v) for v in self._dense_model.embed(texts)]

    def _embed_sparse_sync(self, texts: list[str]) -> list[SparseVectorData]:
        if self._sparse_model is None:
            self._sparse_model = self._sparse_loader()
        return [
            SparseVectorData(
                indices=_to_int_list(item.indices),
                values=_to_float_list(item.values),
            )
            for item in self._sparse_model.embed(texts)
        ]

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_dense_sync, texts)

    async def embed_sparse(self, texts: list[str]) -> list[SparseVectorData]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sparse_sync, texts)


class ONNXRerankerProvider:
    """Lazy ONNX Cross-Encoder reranker scoring ParentChunk candidates."""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        model_loader: Callable[[], Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._model_loader = model_loader or self._default_loader
        self._model: Any | None = None

    def _default_loader(self) -> Any:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(model_name=self._model_name)

    def _rerank_sync(self, query: str, docs: list[str]) -> list[float]:
        if self._model is None:
            self._model = self._model_loader()
        return [float(s) for s in self._model.rerank(query, docs)]

    async def rerank(
        self,
        query: str,
        parents: list[ParentChunkRecord],
        top_k: int = 5,
    ) -> list[ScoredParent]:
        if not parents:
            return []
        docs = [parent.content for parent in parents]
        scores = await asyncio.to_thread(self._rerank_sync, query, docs)
        paired = [
            ScoredParent(parent=parent, score=score)
            for parent, score in zip(parents, scores, strict=True)
        ]
        paired.sort(key=lambda item: item.score, reverse=True)
        return paired[:top_k]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


class FakeEmbeddingProvider:
    """Deterministic lightweight embedding fake for offline TDD without ONNX downloads."""

    def __init__(self, dense_dim: int = 16) -> None:
        self._dense_dim = dense_dim

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dense_dim
            tokens = _tokenize(text) or ["empty"]
            for tok in tokens:
                digest = hashlib.sha256(tok.encode("utf-8")).digest()
                bucket = digest[0] % self._dense_dim
                sign = 1.0 if digest[1] % 2 == 0 else -1.0
                vec[bucket] += sign
            norm = sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    async def embed_sparse(self, texts: list[str]) -> list[SparseVectorData]:
        results: list[SparseVectorData] = []
        for text in texts:
            counts: dict[int, float] = {}
            for tok in _tokenize(text):
                idx = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16) % 100_000
                counts[idx] = counts.get(idx, 0.0) + 1.0
            ordered = sorted(counts.items())
            results.append(
                SparseVectorData(
                    indices=[idx for idx, _ in ordered],
                    values=[val for _, val in ordered],
                )
            )
        return results


class FakeRerankerProvider:
    """Deterministic lexical-overlap reranker fake for workflow unit tests."""

    async def rerank(
        self,
        query: str,
        parents: list[ParentChunkRecord],
        top_k: int = 5,
    ) -> list[ScoredParent]:
        q_tokens = set(_tokenize(query))
        scored: list[ScoredParent] = []
        for parent in parents:
            doc_tokens = set(_tokenize(f"{parent.breadcrumb} {parent.content}"))
            overlap = len(q_tokens & doc_tokens)
            score = overlap / max(len(q_tokens), 1)
            scored.append(ScoredParent(parent=parent, score=float(score)))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
