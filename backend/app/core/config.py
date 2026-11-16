"""Application configuration backed by Pydantic BaseSettings.

Centralizes environment variables so local vs. Docker Compose overrides
(`QDRANT_URL`, `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL`, `DEEPSEEK_API_KEY`)
never scatter `os.getenv()` calls across modules.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed runtime settings for Sovereign Agentic RAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "sovereign-agentic-rag"
    app_version: str = "0.1.0"

    default_inference_mode: Literal["LOCAL_ONPREM", "CLOUD_API"] = "LOCAL_ONPREM"

    # LOCAL_ONPREM defaults tuned for 16GB RAM edge workstations
    local_llm_base_url: str = "http://localhost:11434/v1"
    local_llm_model: str = "hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M"
    local_llm_api_key: str = "ollama"

    # CLOUD_API (DeepSeek Prefix Context Caching)
    deepseek_api_base_url: str = "https://api.deepseek.com"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"

    # Local ONNX FastEmbed models
    embedding_dense_model: str = "BAAI/bge-m3"
    embedding_sparse_model: str = "Qdrant/bm25"
    embedding_dense_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Persistence endpoints
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "sovereign_rag_chunks"
    sqlite_url: str = "sqlite+aiosqlite:///./sovereign_rag.db"

    # Bounded agent loop invariant
    max_retries: int = 2

    # CORS origins for Next.js App Router UI
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
