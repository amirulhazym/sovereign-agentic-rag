"""Contract tests for POST /api/v1/chat/stream SSE events, /ready probe, and history."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.agents.state import GraderOutput, RewriterOutput, RouterOutput
from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError
from app.main import create_app
from app.providers.base import ChatMessage, FakeLLMProvider, StreamChunk
from app.providers.embeddings import FakeEmbeddingProvider, FakeRerankerProvider
from app.repositories.qdrant_repo import InMemoryVectorRepository


def _parse_sse_frames(raw_body: str) -> list[tuple[str, dict[str, Any]]]:
    frames: list[tuple[str, dict[str, Any]]] = []
    for block in raw_body.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_json: dict[str, Any] = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_json = json.loads(line.removeprefix("data:").strip())
        if event_name:
            frames.append((event_name, data_json))
    return frames


def _build_app(
    tmp_path: Path,
    *,
    llm_providers: dict[str, Any] | None = None,
    vector_repo: InMemoryVectorRepository | None = None,
):
    db_file = tmp_path / "sse_test.db"
    settings = Settings(
        _env_file=None,
        sqlite_url=f"sqlite+aiosqlite:///{db_file.as_posix()}",
    )
    vec = vector_repo or InMemoryVectorRepository()
    embedder = FakeEmbeddingProvider(dense_dim=16)
    reranker = FakeRerankerProvider()
    return create_app(
        settings=settings,
        vector_repo=vec,
        embedding_provider=embedder,
        reranker_provider=reranker,
        llm_providers=llm_providers,
    )


def test_ready_probe_reports_ok_and_503_when_dependency_unhealthy(tmp_path: Path) -> None:
    """Verify /ready checks SQLite, Qdrant, and active LLM provider reachability."""
    healthy_llm = FakeLLMProvider()
    app = _build_app(
        tmp_path,
        llm_providers={"LOCAL_ONPREM": healthy_llm, "CLOUD_API": healthy_llm},
    )
    with TestClient(app) as client:
        res = client.get("/ready")
        assert res.status_code == 200
        assert res.json()["status"] == "ready"
        assert res.json()["checks"] == {"sqlite": "ok", "qdrant": "ok", "llm": "ok"}

    class _DownVectorRepo(InMemoryVectorRepository):
        async def ping(self) -> bool:
            return False

    down_app = _build_app(
        tmp_path,
        llm_providers={"LOCAL_ONPREM": healthy_llm},
        vector_repo=_DownVectorRepo(),
    )
    with TestClient(down_app) as client:
        res_down = client.get("/ready")
        assert res_down.status_code == 503
        assert res_down.json()["checks"]["qdrant"] == "unavailable"


def test_chat_stream_returns_404_for_missing_collection(tmp_path: Path) -> None:
    """Verify invalid collection_id fails with standard 404 JSON envelope before SSE opens."""
    llm = FakeLLMProvider()
    app = _build_app(tmp_path, llm_providers={"LOCAL_ONPREM": llm, "CLOUD_API": llm})
    with TestClient(app) as client:
        res = client.post(
            "/api/v1/chat/stream",
            json={
                "collection_id": "missing_ws",
                "question": "Hello?",
                "inference_mode": "LOCAL_ONPREM",
            },
        )
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "COLLECTION_NOT_FOUND"


def test_chat_stream_emits_typed_sse_headers_events_and_persists_turns(tmp_path: Path) -> None:
    """Verify 5 typed SSE events, anti-buffering headers, and multi-turn SQLite history."""
    cloud_llm = FakeLLMProvider(
        structured_responses=[
            RouterOutput(route="single_collection", detected_language="mixed"),
            RewriterOutput(queries=["Klausa 4.2 overtime public holiday"], reasoning="ok"),
            GraderOutput(
                is_sufficient=True,
                approved_parent_ids=["hr_employment_and_overtime_policy_my_p2_s0"],
                reason="Matched Clause 4.2",
            ),
            # Turn 2 direct greeting
            RouterOutput(route="direct", detected_language="ms"),
        ],
        stream_tokens=[
            "Bayaran ialah 2.0x ",
            "[hr_employment_and_overtime_policy_my:p2].",
        ],
        cache_hit_tokens=1280,
    )
    app = _build_app(
        tmp_path,
        llm_providers={"LOCAL_ONPREM": FakeLLMProvider(), "CLOUD_API": cloud_llm},
    )

    with TestClient(app) as client:
        seed_res = client.post("/api/v1/collections/seed")
        assert seed_res.status_code == 202

        res = client.post(
            "/api/v1/chat/stream",
            json={
                "collection_id": "malaysian_enterprise_policy",
                "question": "Apa prosedur tuntutan OT kalau kerja public holiday?",
                "inference_mode": "CLOUD_API",
            },
        )
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        assert res.headers["cache-control"] == "no-cache, no-transform"
        assert res.headers["x-accel-buffering"] == "no"

        frames = _parse_sse_frames(res.text)
        event_names = [name for name, _ in frames]
        assert "node_status" in event_names
        assert "token" in event_names
        assert "citations" in event_names
        assert event_names[-1] == "done"

        citations_payload = next(data for name, data in frames if name == "citations")
        assert len(citations_payload["sources"]) == 1
        assert (
            citations_payload["sources"][0]["citation_key"]
            == "[hr_employment_and_overtime_policy_my:p2]"
        )

        done_payload = next(data for name, data in frames if name == "done")
        session_id = done_payload["session_id"]
        assert done_payload["mode"] == "CLOUD_API"
        assert done_payload["cache_hit_tokens"] == 1280

        # Verify history persisted in SQLite and accessible via REST
        hist_res = client.get(f"/api/v1/sessions/{session_id}/messages")
        assert hist_res.status_code == 200
        assert len(hist_res.json()) == 2

        sessions_res = client.get("/api/v1/collections/malaysian_enterprise_policy/sessions")
        assert sessions_res.status_code == 200
        assert len(sessions_res.json()) == 1

        # Turn 2 reusing session_id passes Turn 1 messages in chat_history
        res2 = client.post(
            "/api/v1/chat/stream",
            json={
                "collection_id": "malaysian_enterprise_policy",
                "session_id": session_id,
                "question": "Terima kasih!",
                "inference_mode": "CLOUD_API",
            },
        )
        assert res2.status_code == 200
        last_call_roles = [m.role for m in cloud_llm.calls[-1]]
        assert last_call_roles.count("user") == 2
        assert last_call_roles.count("assistant") == 1


def test_chat_stream_emits_error_event_when_provider_fails_midstream(tmp_path: Path) -> None:
    """Verify mid-stream LLM errors serialize as typed `event: error` frames cleanly."""

    class _FailingLLMProvider:
        async def ping(self) -> bool:
            return False

        async def generate_structured(
            self, messages: list[ChatMessage], schema: type[Any]
        ) -> Any:
            raise ProviderUnavailableError(
                "OLLAMA_UNAVAILABLE", "Local Ollama daemon unreachable on port 11434."
            )

        async def stream_answer(
            self, messages: list[ChatMessage]
        ) -> AsyncIterator[StreamChunk]:
            raise ProviderUnavailableError("OLLAMA_UNAVAILABLE", "unreachable")
            yield StreamChunk()

    app = _build_app(
        tmp_path,
        llm_providers={"LOCAL_ONPREM": _FailingLLMProvider(), "CLOUD_API": FakeLLMProvider()},
    )
    with TestClient(app) as client:
        client.post("/api/v1/collections", json={"id": "col_a", "name": "Col A"})
        res = client.post(
            "/api/v1/chat/stream",
            json={
                "collection_id": "col_a",
                "question": "Check NVMe status",
                "inference_mode": "LOCAL_ONPREM",
            },
        )
    assert res.status_code == 200
    frames = _parse_sse_frames(res.text)
    assert frames[-1][0] == "error"
    assert frames[-1][1]["code"] == "OLLAMA_UNAVAILABLE"
    assert frames[-1][1]["recoverable"] is True


async def test_chat_service_aborts_stream_when_client_disconnects(tmp_path: Path) -> None:
    """Verify request.is_disconnected() stops streaming and skips assistant DB save."""
    llm = FakeLLMProvider(
        structured_responses=[RouterOutput(route="direct", detected_language="en")],
        stream_tokens=["Token 1 ", "Token 2 ", "Token 3 "],
    )
    app = _build_app(tmp_path, llm_providers={"LOCAL_ONPREM": llm, "CLOUD_API": llm})
    with TestClient(app) as client:
        client.post("/api/v1/collections", json={"id": "col_a", "name": "Col A"})
        chat_service = app.state.chat_service

        from app.schemas.chat import ChatStreamRequest

        req = ChatStreamRequest(
            collection_id="col_a",
            question="Hello",
            inference_mode="LOCAL_ONPREM",
        )
        col, sess_id, hist, mode = await chat_service.prepare_stream(req)

        checks = {"calls": 0}

        async def _disconnect_after_first() -> bool:
            checks["calls"] += 1
            return checks["calls"] >= 2

        emitted: list[str] = []
        async for frame in chat_service.stream_events(
            request=req,
            collection=col,
            session_id=sess_id,
            chat_history=hist,
            mode=mode,
            is_disconnected=_disconnect_after_first,
        ):
            emitted.append(frame)

        assert len(emitted) == 1
        assert not any(f.startswith("event: done") for f in emitted)
        messages = await app.state.sqlite_repo.list_session_messages(sess_id)
        # Only user message recorded; incomplete aborted assistant message was not persisted
        assert len(messages) == 1
        assert messages[0].role == "user"
