"""
Qdrant-backed VectorStore (Phase 3).

Local-first connection strategy:
  1. Try the Qdrant server at settings.QDRANT_URL (docker-compose default).
  2. If unreachable, fall back to qdrant-client's embedded *local* mode
     persisted under settings.QDRANT_LOCAL_PATH -- no server required, so
     `uvicorn app.main:app` works on a laptop with no Docker running.

The active mode is logged at startup and visible via /api/v1/health.

Sync qdrant-client calls are wrapped in asyncio.to_thread (see base.py for
why). The same client instance is reused for the process lifetime.
"""
from __future__ import annotations

import asyncio
import logging
import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from app.config import Settings
from app.indexing.base import VectorStore
from app.schemas.models import Chunk

logger = logging.getLogger("voice_rag")

_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "voice-rag-chunk")


def _point_id(chunk_id: str) -> str:
    """Deterministic UUID per chunk id -> upserts are idempotent across rebuilds."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantVectorStore(VectorStore):
    def __init__(self, settings: Settings) -> None:
        self._collection = settings.QDRANT_COLLECTION
        self._client: QdrantClient | None = None
        try:
            client = QdrantClient(url=settings.QDRANT_URL, timeout=2.0)
            client.get_collections()
            self._client = client
            self.mode = "server"
            logger.info("Connected to Qdrant server at %s", settings.QDRANT_URL)
        except Exception as exc:  # noqa: BLE001 - fallback is deliberate
            logger.warning(
                "Qdrant server at %s unreachable (%s); using local mode at %s",
                settings.QDRANT_URL,
                exc,
                settings.QDRANT_LOCAL_PATH,
            )
            self._client = QdrantClient(path=settings.QDRANT_LOCAL_PATH)
            self.mode = "local"

    # -- helpers ---------------------------------------------------------

    def _require_client(self) -> QdrantClient:
        if self._client is None:
            raise RuntimeError("QdrantVectorStore already closed")
        return self._client

    # -- async API -------------------------------------------------------

    async def health_check(self) -> str:
        try:
            await asyncio.to_thread(self._require_client().get_collections)
            return "ok"
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return f"down: {exc}"

    async def ensure_collection(self, dim: int) -> None:
        client = self._require_client()

        def _create() -> None:
            params = models.VectorParams(size=dim, distance=models.Distance.COSINE)
            if client.collection_exists(self._collection):
                info = client.get_collection(self._collection)
                current = info.config.params.vectors.size if info.config else dim
                if current == dim:
                    return
                logger.info(
                    "Recreating collection %s: dim %s -> %d",
                    self._collection,
                    current,
                    dim,
                )
                client.delete_collection(self._collection)
            client.create_collection(
                collection_name=self._collection, vectors_config=params
            )

        await asyncio.to_thread(_create)

    async def upsert_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks/vectors length mismatch: {len(chunks)} vs {len(vectors)}"
            )
        points = [
            models.PointStruct(
                id=_point_id(c.chunk_id),
                vector=np.asarray(vec, dtype=np.float32).tolist(),
                payload=c.model_dump(),
            )
            for c, vec in zip(chunks, vectors)
        ]
        client = self._require_client()
        await asyncio.to_thread(client.upsert, self._collection, points, wait=True)
        return len(points)

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        query_filter: models.Filter | None = None,
    ) -> list[tuple[Chunk, float]]:
        client = self._require_client()
        hits = await asyncio.to_thread(
            client.query_points,
            self._collection,
            query=np.asarray(query_vector, dtype=np.float32).tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        results: list[tuple[Chunk, float]] = []
        for hit in hits.points:
            payload = hit.payload or {}
            chunk = Chunk.model_validate(payload)
            results.append((chunk, float(hit.score)))
        return results

    async def count_points(self) -> int:
        client = self._require_client()
        count = await asyncio.to_thread(client.count, self._collection, exact=True)
        return int(count.count)

    async def close(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            await asyncio.to_thread(client.close)
