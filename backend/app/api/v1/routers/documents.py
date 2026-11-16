"""REST router for asynchronous document upload and ingestion status polling."""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Response, UploadFile, status

from app.core.dependencies import get_ingestion_service, get_sqlite_repo
from app.repositories.sqlite_repo import SQLiteRepository
from app.schemas.workspace import DocumentRead
from app.services.ingestion import DocumentIngestionService, UploadDispatchResult

router = APIRouter(prefix="/api/v1", tags=["documents"])


@router.post(
    "/collections/{collection_id}/documents",
    response_model=UploadDispatchResult,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload document for async background ingestion",
)
async def upload_document(
    collection_id: str,
    background_tasks: BackgroundTasks,
    response: Response,
    file: Annotated[UploadFile, File(...)],
    ingestion_service: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
) -> UploadDispatchResult:
    """Validate & SHA-256 deduplicate upload, dispatching CPU parsing via BackgroundTasks."""
    file_bytes = await file.read()
    filename = file.filename or "uploaded_document.txt"
    result = await ingestion_service.prepare_upload(
        collection_id=collection_id,
        filename=filename,
        file_bytes=file_bytes,
    )
    if result.deduplicated:
        response.status_code = status.HTTP_200_OK
        return result

    background_tasks.add_task(
        ingestion_service.process_document,
        doc_id=result.document.doc_id,
        collection_id=collection_id,
        filename=filename,
        file_bytes=file_bytes,
    )
    return result


@router.get(
    "/documents/{doc_id}/status",
    response_model=DocumentRead,
    summary="Poll document ingestion lifecycle status",
)
async def get_document_status(
    doc_id: str,
    sqlite_repo: Annotated[SQLiteRepository, Depends(get_sqlite_repo)],
) -> DocumentRead:
    """Return current PENDING / INDEXING / READY / FAILED status for an uploaded document."""
    return await sqlite_repo.get_document(doc_id)
