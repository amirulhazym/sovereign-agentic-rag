"""FastAPI application factory with zero import-time side effects."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient

from app.api.v1.routers.chat import router as chat_router
from app.api.v1.routers.collections import router as collections_router
from app.api.v1.routers.documents import router as documents_router
from app.api.v1.routers.health import router as health_router
from app.core.config import Settings
from app.core.exceptions import register_exception_handlers
from app.providers.base import LLMProvider
from app.providers.cloud_deepseek import CloudDeepSeekLLMAdapter
from app.providers.embeddings import (
    EmbeddingProvider,
    FastEmbedProvider,
    ONNXRerankerProvider,
    RerankerProvider,
)
from app.providers.local_onprem import LocalOnPremLLMAdapter
from app.repositories.qdrant_repo import QdrantHybridRepository, VectorRepository
from app.repositories.sqlite_repo import SQLiteRepository
from app.services.chat_service import ChatService
from app.services.ingestion import DocumentIngestionService
from app.services.seeder import SeederService, resolve_seed_dir

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    sqlite_repo: SQLiteRepository | None = None,
    vector_repo: VectorRepository | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    reranker_provider: RerankerProvider | None = None,
    llm_providers: dict[str, LLMProvider] | None = None,
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
            qdrant_client = AsyncQdrantClient(
                url=resolved_settings.qdrant_url,
                check_compatibility=False,
            )
            resolved_vector_repo = QdrantHybridRepository(
                client=qdrant_client,
                collection_name=resolved_settings.qdrant_collection_name,
                dense_dim=resolved_settings.embedding_dense_dim,
            )

        resolved_embedder = embedding_provider or FastEmbedProvider(
            dense_model_name=resolved_settings.embedding_dense_model,
            sparse_model_name=resolved_settings.embedding_sparse_model,
        )
        resolved_reranker = reranker_provider or ONNXRerankerProvider(
            model_name=resolved_settings.reranker_model,
        )

        http_clients: list[httpx.AsyncClient] = []
        if llm_providers is not None:
            resolved_llms = llm_providers
        else:
            local_client = httpx.AsyncClient(
                base_url=resolved_settings.local_llm_base_url,
                timeout=120.0,
            )
            cloud_client = httpx.AsyncClient(
                base_url=resolved_settings.deepseek_api_base_url,
                timeout=120.0,
            )
            http_clients.extend([local_client, cloud_client])
            resolved_llms = {
                "LOCAL_ONPREM": LocalOnPremLLMAdapter(
                    client=local_client,
                    model=resolved_settings.local_llm_model,
                    api_key=resolved_settings.local_llm_api_key,
                ),
                "CLOUD_API": CloudDeepSeekLLMAdapter(
                    client=cloud_client,
                    api_key=resolved_settings.deepseek_api_key,
                    model=resolved_settings.deepseek_model,
                ),
            }

        app.state.sqlite_repo = db_repo
        app.state.vector_repo = resolved_vector_repo
        app.state.embedding_provider = resolved_embedder
        app.state.reranker_provider = resolved_reranker
        app.state.llm_providers = resolved_llms

        ingestion_service = DocumentIngestionService(
            sqlite_repo=db_repo,
            vector_repo=resolved_vector_repo,
            embedding_provider=resolved_embedder,
        )
        app.state.ingestion_service = ingestion_service
        app.state.seeder_service = SeederService(
            sqlite_repo=db_repo,
            ingestion_service=ingestion_service,
            seed_dir=resolve_seed_dir(resolved_settings.seed_collections_dir),
        )
        app.state.chat_service = ChatService(
            sqlite_repo=db_repo,
            vector_repo=resolved_vector_repo,
            embedding_provider=resolved_embedder,
            reranker_provider=resolved_reranker,
            llm_providers=resolved_llms,
            default_inference_mode=resolved_settings.default_inference_mode,
            max_retries=resolved_settings.max_retries,
        )

        try:
            yield
        finally:
            await db_repo.close()
            if qdrant_client is not None:
                await qdrant_client.close()
            for client in http_clients:
                await client.aclose()

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
    app.include_router(chat_router)
    return app


app = create_app()
