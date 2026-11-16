"""Contract tests for app factory, Settings defaults, health check, and error envelopes."""

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.exceptions import CollectionNotFoundError
from app.main import create_app


def test_settings_defaults_enforce_qwen3_8_and_bounded_retries() -> None:
    """Verify safe 16GB RAM Qwen3.8 defaults and strict max_retries=2 guardrail."""
    settings = Settings(_env_file=None)
    assert settings.local_llm_model == "hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M"
    assert settings.max_retries == 2
    assert settings.default_inference_mode == "LOCAL_ONPREM"
    assert settings.qdrant_collection_name == "sovereign_rag_chunks"


def test_health_endpoint_returns_ok_and_request_id() -> None:
    """Verify /health returns 200 OK without touching external vector or LLM services."""
    app = create_app(settings=Settings(_env_file=None, sqlite_url="sqlite+aiosqlite:///:memory:"))
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "sovereign-agentic-rag"
    assert "X-Request-ID" in response.headers


def test_domain_exception_returns_uniform_error_envelope() -> None:
    """Verify AppError subclasses serialize into standard JSON error envelopes."""
    app = create_app(settings=Settings(_env_file=None, sqlite_url="sqlite+aiosqlite:///:memory:"))

    @app.get("/_test/missing-collection")
    async def trigger_missing() -> None:
        raise CollectionNotFoundError("unknown_ws")

    with TestClient(app) as client:
        response = client.get("/_test/missing-collection")

    assert response.status_code == 404
    body = response.json()
    assert set(body.keys()) == {"error"}
    assert body["error"]["code"] == "COLLECTION_NOT_FOUND"
    assert "unknown_ws" in body["error"]["message"]
    assert body["error"]["request_id"] == response.headers["X-Request-ID"]


def test_unknown_route_uses_uniform_error_envelope() -> None:
    """Verify HTTP 404 errors are wrapped in the same standard error envelope."""
    app = create_app(settings=Settings(_env_file=None, sqlite_url="sqlite+aiosqlite:///:memory:"))
    with TestClient(app) as client:
        response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "HTTP_404"
    assert body["error"]["request_id"] == response.headers["X-Request-ID"]
