"""Uniform Pydantic v2 error envelope schemas for REST error responses."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class ErrorDetail(BaseModel):
    """Machine-readable error payload nested under `error`."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    detail: Any | None = None
    request_id: str


class ErrorEnvelope(BaseModel):
    """Top-level JSON error wrapper returned by all HTTP exception handlers."""

    model_config = ConfigDict(frozen=True)

    error: ErrorDetail
