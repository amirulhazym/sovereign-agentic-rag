"""Health liveness probe router."""

from fastapi import APIRouter, Request

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
