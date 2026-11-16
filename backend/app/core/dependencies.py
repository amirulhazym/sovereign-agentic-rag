"""FastAPI Dependency Injection providers wired from app.state."""

from fastapi import Request

from app.core.config import Settings
from app.providers.embeddings import EmbeddingProvider
from app.repositories.qdrant_repo import VectorRepository
from app.repositories.sqlite_repo import SQLiteRepository
from app.services.ingestion import DocumentIngestionService
from app.services.seeder import SeederService


def get_settings(request: Request) -> Settings:
    """Return application Settings stored during create_app()."""
    return request.app.state.settings


def get_sqlite_repo(request: Request) -> SQLiteRepository:
    """Return initialized async SQLiteRepository from app.state."""
    return request.app.state.sqlite_repo


def get_vector_repo(request: Request) -> VectorRepository:
    """Return VectorRepository implementation (Qdrant or test fake)."""
    return request.app.state.vector_repo


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    """Return EmbeddingProvider implementation (FastEmbed ONNX or test fake)."""
    return request.app.state.embedding_provider


def get_ingestion_service(request: Request) -> DocumentIngestionService:
    """Return DocumentIngestionService from app.state."""
    return request.app.state.ingestion_service


def get_seeder_service(request: Request) -> SeederService:
    """Return SeederService from app.state."""
    return request.app.state.seeder_service
