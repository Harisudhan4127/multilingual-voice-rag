"""
Sentence-based chunking: groups N sentences per chunk, using simple regex
sentence splitting (no heavy NLP dependency needed for this).
"""
from __future__ import annotations

import re

from app.chunking.base import Chunker
from app.schemas.models import Chunk, Document

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


class SentenceChunker(Chunker):
    strategy_name = "sentence"

    def __init__(self, sentences_per_chunk: int = 3, overlap_sentences: int = 1) -> None:
        if overlap_sentences >= sentences_per_chunk:
            raise ValueError("overlap_sentences must be smaller than sentences_per_chunk")
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = split_sentences(document.text)
        if not sentences:
            return []

        step = self.sentences_per_chunk - self.overlap_sentences
        texts: list[str] = []
        i = 0
        while i < len(sentences):
            group = sentences[i : i + self.sentences_per_chunk]
            texts.append(" ".join(group))
            if i + self.sentences_per_chunk >= len(sentences):
                break
            i += step

        return self._build_chunks(document, texts)
