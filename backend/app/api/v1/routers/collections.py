"""REST router for isolated multi-tenant Collection workspaces."""

from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.core.dependencies import get_sqlite_repo
from app.repositories.sqlite_repo import SQLiteRepository
from app.schemas.workspace import CollectionCreate, CollectionRead, DocumentRead

router = APIRouter(prefix="/api/v1/collections", tags=["collections"])


@router.get("", response_model=list[CollectionRead], summary="List all collections")
async def list_collections(
    sqlite_repo: Annotated[SQLiteRepository, Depends(get_sqlite_repo)],
) -> list[CollectionRead]:
    """List every Collection workspace alongside its document count."""
    return await sqlite_repo.list_collections()


@router.post(
    "",
    response_model=CollectionRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new isolated collection",
)
async def create_collection(
    payload: CollectionCreate,
    sqlite_repo: Annotated[SQLiteRepository, Depends(get_sqlite_repo)],
) -> CollectionRead:
    """Register a new tenant Collection workspace."""
    return await sqlite_repo.create_collection(payload)


@router.get(
    "/{collection_id}/documents",
    response_model=list[DocumentRead],
    summary="List documents inside a collection",
)
async def list_collection_documents(
    collection_id: str,
    sqlite_repo: Annotated[SQLiteRepository, Depends(get_sqlite_repo)],
) -> list[DocumentRead]:
    """List DocumentRecords belonging strictly to `collection_id`."""
    return await sqlite_repo.list_collection_documents(collection_id)
