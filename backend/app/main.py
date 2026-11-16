"""FastAPI application factory with zero import-time side effects."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient

from app.api.v1.routers.collections import router as collections_router
from app.api.v1.routers.documents import router as documents_router
from app.api.v1.routers.health import router as health_router
from app.core.config import Settings
from app.core.exceptions import register_exception_handlers
from app.providers.embeddings import EmbeddingProvider, FastEmbedProvider
from app.repositories.qdrant_repo import QdrantHybridRepository, VectorRepository
from app.repositories.sqlite_repo import SQLiteRepository
from app.services.ingestion import DocumentIngestionService

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    sqlite_repo: SQLiteRepository | None = None,
    vector_repo: VectorRepository | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> FastAPI:
    """Create and configure a FastAPI instance with uniform error handling and lifespan."""
    resolved_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Wire stateful storage and lazy providers inside lifespan (no import side effects)."""
        db_repo = sqlite_repo or SQLiteRepository(resolved_settings.sqlite_url)
        await db_repo.init_schema()

        qdrant_client: AsyncQdrantClient | None = None
        if vector_repo is not None:
            resolved_vector_repo = vector_repo
        else:
            qdrant_client = AsyncQdrantClient(url=resolved_settings.qdrant_url)
            resolved_vector_repo = QdrantHybridRepository(
                client=qdrant_client,
                collection_name=resolved_settings.qdrant_collection_name,
                dense_dim=resolved_settings.embedding_dense_dim,
            )

        resolved_embedder = embedding_provider or FastEmbedProvider(
            dense_model_name=resolved_settings.embedding_dense_model,
            sparse_model_name=resolved_settings.embedding_sparse_model,
        )

        app.state.sqlite_repo = db_repo
        app.state.vector_repo = resolved_vector_repo
        app.state.embedding_provider = resolved_embedder
        app.state.ingestion_service = DocumentIngestionService(
            sqlite_repo=db_repo,
            vector_repo=resolved_vector_repo,
            embedding_provider=resolved_embedder,
        )

        try:
            yield
        finally:
            await db_repo.close()
            if qdrant_client is not None:
                await qdrant_client.close()

    app = FastAPI(
        title="Sovereign Agentic RAG API",
        version=resolved_settings.app_version,
        description=(
            "Zero-cost, dual-mode (LOCAL_ONPREM Qwen3.8 / CLOUD_API DeepSeek) "
            "Agentic RAG platform with Qdrant Hybrid RRF and typed SSE streaming."
        ),
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def attach_request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or f"req_{uuid4().hex[:12]}"
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(collections_router)
    app.include_router(documents_router)
    return app


app = create_app()
