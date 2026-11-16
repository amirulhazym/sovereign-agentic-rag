"""Domain exceptions and FastAPI error envelope handlers."""

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas.errors import ErrorDetail, ErrorEnvelope


class AppError(Exception):
    """Base domain exception mapped to a consistent HTTP status and JSON envelope."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        detail: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.detail = detail


class CollectionNotFoundError(AppError):
    """Raised when a tenant collection_id does not exist."""

    def __init__(self, collection_id: str) -> None:
        super().__init__(
            code="COLLECTION_NOT_FOUND",
            message=f"Collection '{collection_id}' does not exist.",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class CollectionConflictError(AppError):
    """Raised when attempting to create a collection with an existing id."""

    def __init__(self, collection_id: str) -> None:
        super().__init__(
            code="COLLECTION_ALREADY_EXISTS",
            message=f"Collection '{collection_id}' already exists.",
            status_code=status.HTTP_409_CONFLICT,
        )


class DocumentNotFoundError(AppError):
    """Raised when a document id cannot be found."""

    def __init__(self, doc_id: str) -> None:
        super().__init__(
            code="DOCUMENT_NOT_FOUND",
            message=f"Document '{doc_id}' does not exist.",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class UnsupportedDocumentTypeError(AppError):
    """Raised when an uploaded file extension is outside (.pdf, .md, .txt)."""

    def __init__(self, filename: str) -> None:
        super().__init__(
            code="UNSUPPORTED_MEDIA_TYPE",
            message=f"Unsupported file type for '{filename}'. Allowed: .pdf, .md, .txt.",
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )


class ProviderUnavailableError(AppError):
    """Raised when an upstream LLM or vector provider cannot be reached."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(
            code=code,
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _envelope_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    detail: Any | None = None,
) -> JSONResponse:
    request_id = _get_request_id(request)
    payload = ErrorEnvelope(
        error=ErrorDetail(
            code=code,
            message=message,
            detail=detail,
            request_id=request_id,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(),
        headers={"X-Request-ID": request_id},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach uniform JSON error envelope handlers to the FastAPI instance."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return _envelope_response(
            request=request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            detail=exc.detail,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return _envelope_response(
            request=request,
            status_code=exc.status_code,
            code=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _envelope_response(
            request=request,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="VALIDATION_ERROR",
            message="Request validation failed.",
            detail=exc.errors(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        return _envelope_response(
            request=request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected internal error occurred.",
        )
