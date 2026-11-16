"""Idempotent SeederService populating Collection A & Collection B demo knowledge bases."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.core.exceptions import CollectionConflictError
from app.repositories.sqlite_repo import SQLiteRepository
from app.schemas.workspace import CollectionCreate
from app.services.ingestion import DocumentIngestionService


@dataclass(frozen=True)
class _SeedCollectionSpec:
    id: str
    name: str
    description: str


_DEMO_COLLECTIONS: tuple[_SeedCollectionSpec, ...] = (
    _SeedCollectionSpec(
        id="it_infra_sops",
        name="Collection A: Enterprise IT & Server Infrastructure SOPs",
        description=(
            "Technical hardware specification matrices, NVMe storage maintenance SOPs, "
            "and Tier-1/Tier-2 disaster recovery failover runbooks."
        ),
    ),
    _SeedCollectionSpec(
        id="malaysian_enterprise_policy",
        name="Collection B: Malaysian Enterprise Policy & Compliance",
        description=(
            "Bilingual English, Bahasa Melayu, and BM-EN workplace policies covering HR "
            "overtime claims, outstation allowances, and PDPA 2010/2024 compliance."
        ),
    ),
)


class SeedReport(BaseModel):
    """Summary returned by POST /api/v1/collections/seed."""

    model_config = ConfigDict(frozen=True)

    collections_created: int
    documents_ingested: int
    documents_skipped_duplicate: int
    collection_ids: list[str]


def resolve_seed_dir(configured_path: str = "") -> Path:
    """Locate data/seed_collections across local workspace and Docker container layouts."""
    if configured_path:
        return Path(configured_path)
    repo_candidate = Path(__file__).resolve().parents[3] / "data" / "seed_collections"
    if repo_candidate.exists():
        return repo_candidate
    return Path("/app/data/seed_collections")


class SeederService:
    """Seeds pre-curated Collection A & B documents idempotently via SHA-256 checks."""

    def __init__(
        self,
        *,
        sqlite_repo: SQLiteRepository,
        ingestion_service: DocumentIngestionService,
        seed_dir: Path,
    ) -> None:
        self._sqlite = sqlite_repo
        self._ingestion = ingestion_service
        self._seed_dir = seed_dir

    async def seed_all(self) -> SeedReport:
        """Register demo collections and ingest any unindexed seed files synchronously."""
        collections_created = 0
        documents_ingested = 0
        documents_skipped = 0
        collection_ids: list[str] = []

        for spec in _DEMO_COLLECTIONS:
            collection_ids.append(spec.id)
            try:
                await self._sqlite.create_collection(
                    CollectionCreate(
                        id=spec.id,
                        name=spec.name,
                        description=spec.description,
                        is_seeded=True,
                    )
                )
                collections_created += 1
            except CollectionConflictError:
                pass

            folder = self._seed_dir / spec.id
            if not folder.is_dir():
                continue

            for file_path in sorted(folder.iterdir()):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in {".pdf", ".md", ".txt"}:
                    continue

                file_bytes = file_path.read_bytes()
                dispatch = await self._ingestion.prepare_upload(
                    collection_id=spec.id,
                    filename=file_path.name,
                    file_bytes=file_bytes,
                    doc_id=file_path.stem,
                )
                if dispatch.deduplicated:
                    documents_skipped += 1
                    continue

                await self._ingestion.process_document(
                    doc_id=dispatch.document.doc_id,
                    collection_id=spec.id,
                    filename=file_path.name,
                    file_bytes=file_bytes,
                )
                documents_ingested += 1

        return SeedReport(
            collections_created=collections_created,
            documents_ingested=documents_ingested,
            documents_skipped_duplicate=documents_skipped,
            collection_ids=collection_ids,
        )
