"""
Centralized application configuration.

All configuration must be read from here. No module outside this file
should read os.environ directly. This is what makes provider swapping
(mock -> sarvam, mock -> real LLM, synthetic -> MSMARCO-XI) a config
change instead of a code change.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    APP_ENV: Literal["development", "test", "production"] = "development"
    APP_NAME: str = "voice-rag"
    LOG_LEVEL: str = "INFO"

    # --- Providers (the swap points) ---
    STT_PROVIDER: Literal["mock", "sarvam"] = "mock"
    LLM_PROVIDER: Literal["mock", "real"] = "mock"
    VECTOR_PROVIDER: Literal["qdrant"] = "qdrant"
    DATASET_PROVIDER: Literal["synthetic", "msmarco_xi"] = "synthetic"

    # --- Sarvam STT ---
    SARVAM_API_KEY: str = ""
    SARVAM_BASE_URL: str = "https://api.sarvam.ai"
    SARVAM_STT_PATH: str = "/speech-to-text"
    SARVAM_STT_MODEL: str = "saarika:v2"

    # --- Real LLM ---
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = ""
    # Ask an OpenAI-compatible API for strict JSON output when supported.
    LLM_JSON_MODE: bool = True

    # --- MSMARCO-XI dataset ---
    DATASET_LANGUAGE: str = "hi"  # dataset config: as bn gu hi kn ml mr ne or pa sa ta te ur
    MSMARCO_SPLIT: str = "validation"  # 'train' holds millions of rows
    MSMARCO_MAX_DOCS: int = 500  # cap indexed passages for local runs

    # --- Vector DB ---
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "voice_rag_chunks"
    QDRANT_LOCAL_PATH: str = "./data/qdrant_local"  # used when server unreachable

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_FALLBACK_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    # Last-resort offline embedder (deterministic hashing, degraded quality).
    # Used only when neither embedding model can be downloaded/loaded.
    ALLOW_HASHING_EMBEDDER_FALLBACK: bool = True

    # --- Indexing ---
    CHUNKING_STRATEGY: Literal["sentence", "token", "semantic", "parent_child"] = (
        "sentence"
    )
    PROCESSED_CHUNKS_FILE: str = "./data/processed/chunks.jsonl"
    AUTO_INDEX_ON_STARTUP: bool = True

    # --- Retrieval ---
    TOP_K_DENSE: int = 20
    TOP_K_BM25: int = 20
    TOP_K_RERANK: int = 5
    DENSE_WEIGHT: float = 0.65
    BM25_WEIGHT: float = 0.35
    RRF_K: int = 60

    # --- Guardrails ---
    RETRIEVAL_CONFIDENCE_THRESHOLD: float = 0.15
    GROUNDING_THRESHOLD: float = 0.5
    # Off-topic guard: minimum distinct content-word hits against the corpus
    # vocabulary required to consider the query on-topic.
    OFF_TOPIC_MIN_VOCAB_HITS: int = 1

    # --- Reranker ---
    RERANKER_TYPE: Literal["cross_encoder", "lexical"] = "cross_encoder"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Timeouts (seconds) ---
    STT_TIMEOUT_S: float = 10.0
    LLM_TIMEOUT_S: float = 20.0
    RETRIEVAL_TIMEOUT_S: float = 5.0

    # --- Paths ---
    DATA_DIR: str = "./data"

    # --- CORS ---
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Settings are loaded once and cached for the process lifetime."""
    return Settings()
