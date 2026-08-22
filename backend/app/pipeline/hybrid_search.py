"""
Hybrid retrieval (Phase 4 / Section 12):

    Dense top-k  ─┐
                  ├─> weighted Reciprocal Rank Fusion -> top candidates
    BM25 top-k   ─┘

RRF score for a chunk is
    sum over ranklists:  weight * 1 / (rrf_k + rank)      (rank starts at 1)

Weights come from configuration (DENSE_WEIGHT / BM25_WEIGHT / RRF_K) so the
fusion behaviour is tunable without code changes. The two retrievers run in
parallel via asyncio.gather.
"""
from __future__ import annotations

import asyncio
import logging

from app.config import Settings
from app.pipeline.retriever import BaseRetriever, BM25Retriever, DenseRetriever
from app.schemas.models import Chunk

logger = logging.getLogger("voice_rag")


def rrf_fuse(
    dense_results: list[tuple[Chunk, float]],
    bm25_results: list[tuple[Chunk, float]],
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
    top_k: int | None = None,
) -> list[tuple[Chunk, float]]:
    """Fuse two ranked lists with weighted RRF. Pure function; tested directly."""
    scores: dict[str, float] = {}
    best_chunk: dict[str, Chunk] = {}

    def _accumulate(results: list[tuple[Chunk, float]], weight: float) -> None:
        for rank, (chunk, _) in enumerate(results, start=1):
            contribution = weight / (rrf_k + rank)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + contribution
            # Keep the first-seen instance; identical chunk ids carry identical text.
            best_chunk.setdefault(chunk.chunk_id, chunk)

    _accumulate(dense_results, dense_weight)
    _accumulate(bm25_results, bm25_weight)

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = [(best_chunk[cid], score) for cid, score in fused]
    return out[:top_k] if top_k else out


class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def __init__(self, settings: Settings, embedder, vector_store, bm25_store) -> None:
        self._settings = settings
        self.dense = DenseRetriever(embedder, vector_store)
        self.bm25 = BM25Retriever(bm25_store)

    async def search(
        self, query: str, top_k: int | None = None
    ) -> list[tuple[Chunk, float]]:
        s = self._settings
        limit = top_k or max(s.TOP_K_DENSE, s.TOP_K_BM25)

        dense_task = self.dense.search(query, s.TOP_K_DENSE)
        bm25_task = self.bm25.search(query, s.TOP_K_BM25)
        try:
            dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)
        except RuntimeError as exc:
            # If either leg fails entirely, hybrid retrieval cannot produce a
            # trustworthy ranking -- surface a controlled failure to the caller.
            logger.error("Hybrid retrieval failed: %s", exc)
            raise

        return rrf_fuse(
            dense_results,
            bm25_results,
            rrf_k=s.RRF_K,
            dense_weight=s.DENSE_WEIGHT,
            bm25_weight=s.BM25_WEIGHT,
            top_k=limit,
        )
