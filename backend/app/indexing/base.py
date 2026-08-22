"""
Vector store interface (Phase 3).

Configuration decides which implementation is used (VECTOR_PROVIDER=qdrant
today); the retrieval pipeline only ever talks to this interface, so adding
another backend later is a new subclass + a factory branch -- no pipeline
changes.

All methods are async: implementations that wrap sync SDKs translate with
asyncio.to_thread so the API event loop is never blocked.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.schemas.models import Chunk


class VectorStore(ABC):
    mode: str = "abstract"  # e.g. "server" | "local"

    @abstractmethod
    async def health_check(self) -> str:
        """Return 'ok' or a failure description."""
        raise NotImplementedError

    @abstractmethod
    async def ensure_collection(self, dim: int) -> None:
        """Create the collection if missing; recreate if the dim changed."""
        raise NotImplementedError

    @abstractmethod
    async def upsert_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """Upsert chunk payloads + vectors. Returns number of points stored."""
        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        query_filter: object | None = None,
    ) -> list[tuple[Chunk, float]]:
        """KNN search. Returns (chunk, similarity_score) best-first."""
        raise NotImplementedError

    @abstractmethod
    async def count_points(self) -> int:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError
