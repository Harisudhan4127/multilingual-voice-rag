"""
Reranking stage (Phase 4 / Section 13).

After RRF fusion produces top 10-20 candidates, a reranker re-scores each
(query, chunk) pair with a more precise (but more expensive) model and keeps
the best TOP_K_RERANK chunks for generation.

Implementations:
  - CrossEncoderReranker: local cross-encoder/ms-marco-MiniLM-L-6-v2 (~22M
    params). Small on purpose: a huge reranker would make latency benchmarks
    meaningless.
  - LexicalReranker: dependency-free token-overlap scorer. Automatic fallback
    when the cross-encoder model cannot be loaded; also used in tests.
"""
from __future__ import annotations

import asyncio
import logging
import math

from app.config import Settings
from app.indexing.bm25_store import tokenize
from app.schemas.models import Chunk

logger = logging.getLogger("voice_rag")


class BaseReranker:
    kind: str = "base"

    async def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    kind = "cross_encoder"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self.model_name = model_name

    def _score(self, query: str, texts: list[str]) -> list[float]:
        logits = self._model.predict(
            [(query, text) for text in texts],
            batch_size=32,
            show_progress_bar=False,
        )
        # Map raw logits to (0, 1) so scores stay comparable across stages.
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]

    async def rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        texts = [c.text for c, _ in candidates]
        scores = await asyncio.to_thread(self._score, query, texts)
        rescored = sorted(
            zip([c for c, _ in candidates], scores), key=lambda x: x[1], reverse=True
        )
        return [(chunk, float(score)) for chunk, score in rescored[:top_k]]


class LexicalReranker(BaseReranker):
    """
    Fallback scorer: F1 overlap between query tokens and chunk tokens.

    Weaker than a cross-encoder but deterministic, instant and honest about
    what it is -- the health/metrics endpoints report which reranker ran.
    """

    kind = "lexical"

    @staticmethod
    def _f1(query_tokens: set[str], chunk_tokens: list[str]) -> float:
        if not query_tokens or not chunk_tokens:
            return 0.0
        chunk_set = set(chunk_tokens)
        overlap = len(query_tokens & chunk_set)
        precision = overlap / len(chunk_set)
        recall = overlap / len(query_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    async def rerank(self, query, candidates, top_k):
        query_tokens = set(tokenize(query))
        scored = [
            (chunk, self._f1(query_tokens, tokenize(chunk.text)))
            for chunk, _ in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in scored[:top_k]]


def get_reranker(settings: Settings) -> BaseReranker:
    """Load the configured reranker once at startup; fall back gracefully."""
    try:
        reranker = CrossEncoderReranker(settings.RERANKER_MODEL)
        logger.info("Loaded cross-encoder reranker '%s'", settings.RERANKER_MODEL)
        return reranker
    except Exception as exc:  # noqa: BLE001 - fallback is deliberate
        logger.warning(
            "Cross-encoder '%s' unavailable (%s); using lexical reranker fallback",
            settings.RERANKER_MODEL,
            exc,
        )
        return LexicalReranker()
