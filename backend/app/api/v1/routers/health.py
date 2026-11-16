"""Health liveness (/health) and dependency readiness (/ready) probe router."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from app.core.config import Settings
from app.core.dependencies import get_llm_providers, get_settings, get_sqlite_repo, get_vector_repo
from app.providers.base import LLMProvider
from app.repositories.qdrant_repo import VectorRepository
from app.repositories.sqlite_repo import SQLiteRepository
from app.schemas.chat import ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
async def get_health(request: Request) -> dict[str, str]:
    """Return process liveness status without external dependency calls."""
    settings = request.app.state.settings
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def get_ready(
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
    sqlite_repo: Annotated[SQLiteRepository, Depends(get_sqlite_repo)],
    vector_repo: Annotated[VectorRepository, Depends(get_vector_repo)],
    llm_providers: Annotated[dict[str, LLMProvider], Depends(get_llm_providers)],
) -> ReadinessResponse:
    """Verify SQLite, Qdrant, and active default LLM provider reachability."""
    sqlite_ok = False
    try:
        sqlite_ok = await sqlite_repo.ping()
    except Exception:
        sqlite_ok = False

    qdrant_ok = False
    try:
        qdrant_ok = await vector_repo.ping()
    except Exception:
        qdrant_ok = False

    active_llm = llm_providers.get(settings.default_inference_mode)
    llm_ok = False
    if active_llm is not None:
        try:
            llm_ok = await active_llm.ping()
        except Exception:
            llm_ok = False

    checks = {
        "sqlite": "ok" if sqlite_ok else "unavailable",
        "qdrant": "ok" if qdrant_ok else "unavailable",
        "llm": "ok" if llm_ok else "unavailable",
    }
    all_ok = sqlite_ok and qdrant_ok and llm_ok
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if all_ok else "degraded",
        checks=checks,
    )
