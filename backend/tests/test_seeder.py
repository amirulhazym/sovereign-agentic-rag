"""Tests for pre-curated demo seed collections and idempotent SeederService."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.providers.embeddings import FakeEmbeddingProvider
from app.repositories.qdrant_repo import InMemoryVectorRepository


async def test_seed_endpoint_populates_demo_collections_and_is_idempotent(
    tmp_path: Path,
) -> None:
    """Verify POST /api/v1/collections/seed populates Collection A & B cleanly and idempotently."""
    db_file = tmp_path / "seed_test.db"
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

    with TestClient(app) as client:
        first_res = client.post("/api/v1/collections/seed")
        assert first_res.status_code == 202
        first_report = first_res.json()
        assert first_report["collections_created"] == 2
        assert first_report["documents_ingested"] >= 4
        assert first_report["documents_skipped_duplicate"] == 0

        collections = {c["id"]: c for c in client.get("/api/v1/collections").json()}
        assert "it_infra_sops" in collections
        assert "malaysian_enterprise_policy" in collections
        assert collections["it_infra_sops"]["is_seeded"] is True
        assert collections["malaysian_enterprise_policy"]["is_seeded"] is True

        # Second run must skip all already-seeded documents via SHA-256 deduplication
        second_res = client.post("/api/v1/collections/seed")
        assert second_res.status_code == 202
        second_report = second_res.json()
        assert second_report["collections_created"] == 0
        assert second_report["documents_ingested"] == 0
        assert second_report["documents_skipped_duplicate"] >= 4

        # Verify hybrid retrieval pulls distinct domain content within each tenant collection
        it_dense = (await embedder.embed_dense(["ERR_NVME_0x04 hot swap"]))[0]
        it_sparse = (await embedder.embed_sparse(["ERR_NVME_0x04 hot swap"]))[0]
        it_hits = await vector_repo.hybrid_search(
            collection_id="it_infra_sops",
            dense_vector=it_dense,
            sparse_vector=it_sparse,
            top_k=4,
        )
        assert it_hits
        assert any("ERR_NVME_0x04" in h.content for h in it_hits)
        assert all(h.collection_id == "it_infra_sops" for h in it_hits)

        my_dense = (await embedder.embed_dense(["Klausa 4.2 overtime public holiday"]))[0]
        my_sparse = (await embedder.embed_sparse(["Klausa 4.2 overtime public holiday"]))[0]
        my_hits = await vector_repo.hybrid_search(
            collection_id="malaysian_enterprise_policy",
            dense_vector=my_dense,
            sparse_vector=my_sparse,
            top_k=4,
        )
        assert my_hits
        assert any("Klausa 4.2" in h.content or "2.0x" in h.content for h in my_hits)
        assert all(h.collection_id == "malaysian_enterprise_policy" for h in my_hits)
