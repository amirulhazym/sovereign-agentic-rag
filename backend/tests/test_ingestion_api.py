"""Contract tests for collections, async BackgroundTasks ingestion, and SHA-256 dedup."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.providers.embeddings import FakeEmbeddingProvider
from app.repositories.qdrant_repo import InMemoryVectorRepository


def _create_isolated_test_app(tmp_path: Path):
    db_file = tmp_path / "ingestion_test.db"
    settings = Settings(
        _env_file=None,
        sqlite_url=f"sqlite+aiosqlite:///{db_file.as_posix()}",
    )
    vector_repo = InMemoryVectorRepository()
    embedder = FakeEmbeddingProvider(dense_dim=16)
    app = create_app(
        settings=settings,
        vector_repo=vector_repo,
        embedding_provider=embedder,
    )
    return app, vector_repo


def test_collection_crud_and_duplicate_conflict(tmp_path: Path) -> None:
    """Verify creating, listing, and rejecting duplicate collections via REST."""
    app, _ = _create_isolated_test_app(tmp_path)
    with TestClient(app) as client:
        res = client.post(
            "/api/v1/collections",
            json={
                "id": "it_infra_sops",
                "name": "Enterprise IT & Server SOPs",
                "description": "Hardware and runbook documents",
            },
        )
        assert res.status_code == 201
        assert res.json()["id"] == "it_infra_sops"

        dup = client.post(
            "/api/v1/collections",
            json={"id": "it_infra_sops", "name": "Dup"},
        )
        assert dup.status_code == 409
        assert dup.json()["error"]["code"] == "COLLECTION_ALREADY_EXISTS"

        listed = client.get("/api/v1/collections")
        assert listed.status_code == 200
        assert len(listed.json()) == 1


def test_async_document_upload_deduplication_and_status_polling(tmp_path: Path) -> None:
    """Verify 202 async upload, READY status polling, Qdrant indexing, and 200 SHA-256 dedup."""
    app, vector_repo = _create_isolated_test_app(tmp_path)
    markdown_bytes = (
        b"# Storage Maintenance\n"
        b"## NVMe Hot-Swap\n"
        b"Error ERR_NVME_0x04 indicates PCIe link degradation requiring drive reseat.\n"
        b"<!-- page: 2 -->\n"
        b"## Thermal Threshold\n"
        b"Keep chassis inlet temperature <= 35C."
    )

    with TestClient(app) as client:
        client.post(
            "/api/v1/collections",
            json={"id": "it_infra_sops", "name": "IT SOPs"},
        )

        # First upload: HTTP 202 Accepted + BackgroundTasks completion
        upload_res = client.post(
            "/api/v1/collections/it_infra_sops/documents",
            files={"file": ("nvme_sop.md", markdown_bytes, "text/markdown")},
        )
        assert upload_res.status_code == 202
        upload_body = upload_res.json()
        assert upload_body["deduplicated"] is False
        doc_id = upload_body["document"]["doc_id"]

        status_res = client.get(f"/api/v1/documents/{doc_id}/status")
        assert status_res.status_code == 200
        status_body = status_res.json()
        assert status_body["status"] == "READY"
        assert status_body["page_count"] == 2
        assert status_body["chunk_count"] >= 2
        assert status_body["language"] == "en"

        # Duplicate upload with identical bytes -> 200 OK idempotent return
        dup_res = client.post(
            "/api/v1/collections/it_infra_sops/documents",
            files={"file": ("nvme_sop_copy.md", markdown_bytes, "text/markdown")},
        )
        assert dup_res.status_code == 200
        assert dup_res.json()["deduplicated"] is True
        assert dup_res.json()["document"]["doc_id"] == doc_id

        # Collection document listing
        docs_list = client.get("/api/v1/collections/it_infra_sops/documents")
        assert docs_list.status_code == 200
        assert len(docs_list.json()) == 1

        # Verify no duplicate vectors in Qdrant fake
        assert len(vector_repo._items) == status_body["chunk_count"]


def test_unsupported_extension_and_corrupt_pdf_failure_state(tmp_path: Path) -> None:
    """Verify 415 envelope for unsupported files and FAILED status on corrupt PDF bytes."""
    app, _ = _create_isolated_test_app(tmp_path)
    with TestClient(app) as client:
        client.post(
            "/api/v1/collections",
            json={"id": "policy_ws", "name": "Policy"},
        )

        bad_ext = client.post(
            "/api/v1/collections/policy_ws/documents",
            files={"file": ("script.exe", b"MZ...", "application/octet-stream")},
        )
        assert bad_ext.status_code == 415
        assert bad_ext.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"

        corrupt_pdf = client.post(
            "/api/v1/collections/policy_ws/documents",
            files={"file": ("broken.pdf", b"not-a-valid-pdf", "application/pdf")},
        )
        assert corrupt_pdf.status_code == 202
        bad_doc_id = corrupt_pdf.json()["document"]["doc_id"]

        failed_status = client.get(f"/api/v1/documents/{bad_doc_id}/status")
        assert failed_status.status_code == 200
        assert failed_status.json()["status"] == "FAILED"
        assert failed_status.json()["error_message"]


async def test_sqlite_parent_chunks_populated_during_ingestion(tmp_path: Path) -> None:
    """Verify ingested ParentChunks are queryable directly from SQLiteRepository."""
    app, _ = _create_isolated_test_app(tmp_path)
    with TestClient(app) as client:
        client.post(
            "/api/v1/collections",
            json={"id": "policy_ws", "name": "Policy"},
        )
        res = client.post(
            "/api/v1/collections/policy_ws/documents",
            files={
                "file": (
                    "hr.md",
                    b"## Klausa 4.2\nPekerja layak menuntut bayaran 2x pada hari kelepasan am.",
                    "text/markdown",
                )
            },
        )
        doc_id = res.json()["document"]["doc_id"]
        parents = await app.state.sqlite_repo.get_parent_chunks_by_ids([f"{doc_id}_p1_s0"])
        assert len(parents) == 1
        assert parents[0].section_header == "Klausa 4.2"
        assert "bayaran 2x" in parents[0].content
