"""
Retrieval stage (Phase 4): dense + BM25 retrievers behind one interface.

Both retrievers are async so the hybrid search can run them concurrently.
Heavy synchronous work (embedding encode, qdrant call) is pushed to worker
threads via asyncio.to_thread to keep the event loop responsive.
"""
from __future__ import annotations

import asyncio
import logging

from app.indexing.base import VectorStore
from app.indexing.bm25_store import BM25Store
from app.indexing.embeddings import BaseEmbedder
from app.schemas.models import Chunk
from app.utils.cache import TTLCache

logger = logging.getLogger("voice_rag")


class BaseRetriever:
    name: str = "base"

    async def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        raise NotImplementedError


class DenseRetriever(BaseRetriever):
    """Embeds the query with the shared startup embedder, then KNN in Qdrant."""

    name = "dense"

    def __init__(self, embedder: BaseEmbedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._vector_cache = TTLCache(maxsize=256, ttl_seconds=600.0)

    def _embed_cached(self, query: str):
        cached = self._vector_cache.get(query)
        if cached is not None:
            return cached
        vector = self._embedder.embed_one(query)
        self._vector_cache.set(query, vector)
        return vector

    async def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        try:
            vector = await asyncio.to_thread(self._embed_cached, query)
        except Exception as exc:
            logger.error("Dense retrieval embedding failed: %s", exc)
            raise RuntimeError("embedding failure") from exc
        try:
            return await self._vector_store.search(vector, top_k=top_k)
        except Exception as exc:
            logger.error("Dense retrieval vector-store failure: %s", exc)
            raise RuntimeError("vector store unavailable") from exc


class BM25Retriever(BaseRetriever):
    """Wraps the in-memory BM25 index built at startup."""

    name = "bm25"

    def __init__(self, store: BM25Store) -> None:
        self._store = store

    async def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        try:
            return await asyncio.to_thread(self._store.search, query, top_k)
        except Exception as exc:
            logger.error("BM25 retrieval failed: %s", exc)
            raise RuntimeError("bm25 unavailable") from exc

    def __len__(self) -> int:
        return len(self._store)
