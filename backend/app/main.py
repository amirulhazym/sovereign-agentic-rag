"""FastAPI application factory with zero import-time side effects."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routers.health import router as health_router
from app.core.config import Settings
from app.core.exceptions import register_exception_handlers


@asynccontextmanager
async def _default_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize runtime dependencies inside lifespan so module import stays side-effect-free."""
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure a FastAPI instance with uniform error handling and lifespan."""
    resolved_settings = settings or Settings()

    app = FastAPI(
        title="Sovereign Agentic RAG API",
        version=resolved_settings.app_version,
        description=(
            "Zero-cost, dual-mode (LOCAL_ONPREM Qwen3.8 / CLOUD_API DeepSeek) "
            "Agentic RAG platform with Qdrant Hybrid RRF and typed SSE streaming."
        ),
        lifespan=_default_lifespan,
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
    return app


app = create_app()
