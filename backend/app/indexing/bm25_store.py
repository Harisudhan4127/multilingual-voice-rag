"""
Local BM25 index over Chunk objects.

Built early (Phase 2, ahead of the rest of Phase 3's indexing layer)
because the chunking evaluation in Section 8 needs a real retrieval signal
to measure recall/MRR against -- fabricating those numbers is explicitly
disallowed. Dense retrieval joins this in Phase 4 via hybrid RRF fusion;
this module stays focused on BM25 only.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.schemas.models import Chunk

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Store:
    """In-memory BM25 index. Rebuilt from a chunk list; no persistence needed
    for local/dev use since it's cheap to rebuild from data/processed chunks."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        tokenized = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        if self._bm25 is None or not self._chunks:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(zip(self._chunks, scores), key=lambda x: x[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:top_k] if score > 0]

    def __len__(self) -> int:
        return len(self._chunks)
