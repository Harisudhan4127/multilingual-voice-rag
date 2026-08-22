"""
Token-based chunking: splits text into fixed-size windows of whitespace
tokens with configurable overlap. This is a lightweight approximation of a
true subword-tokenizer chunker (kept dependency-free for Phase 2); if a real
tokenizer is later wired in for the chosen embedding model, only this file
needs to change.
"""
from __future__ import annotations

from app.chunking.base import Chunker
from app.schemas.models import Chunk, Document


class TokenChunker(Chunker):
    strategy_name = "token"

    def __init__(self, tokens_per_chunk: int = 60, overlap_tokens: int = 10) -> None:
        if overlap_tokens >= tokens_per_chunk:
            raise ValueError("overlap_tokens must be smaller than tokens_per_chunk")
        self.tokens_per_chunk = tokens_per_chunk
        self.overlap_tokens = overlap_tokens

    def chunk(self, document: Document) -> list[Chunk]:
        tokens = document.text.split()
        if not tokens:
            return []

        step = self.tokens_per_chunk - self.overlap_tokens
        texts: list[str] = []
        i = 0
        while i < len(tokens):
            window = tokens[i : i + self.tokens_per_chunk]
            texts.append(" ".join(window))
            if i + self.tokens_per_chunk >= len(tokens):
                break
            i += step

        return self._build_chunks(document, texts)
