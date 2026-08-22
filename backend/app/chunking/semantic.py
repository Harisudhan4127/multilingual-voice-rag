"""
Semantic chunking: groups consecutive sentences into a chunk as long as they
stay topically similar, and starts a new chunk when similarity drops below a
threshold (a similarity-based greedy grouping, not fixed-size windows).

The similarity function is injected so Phase 3 can swap in real embedding
cosine similarity once the embedding model is loaded at startup, without
touching this file's grouping logic. Until then, a dependency-free lexical
Jaccard similarity over word sets is used as a reasonable local proxy -
still meaningfully "semantic" in the sense that it groups by shared
vocabulary/topic rather than by fixed sentence/token counts.
"""
from __future__ import annotations

from collections.abc import Callable

from app.chunking.base import Chunker
from app.chunking.sentence import split_sentences
from app.schemas.models import Chunk, Document

SimilarityFn = Callable[[str, str], float]


def jaccard_similarity(a: str, b: str) -> float:
    words_a = {w.lower().strip(".,!?;:") for w in a.split()}
    words_b = {w.lower().strip(".,!?;:") for w in b.split()}
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class SemanticChunker(Chunker):
    strategy_name = "semantic"

    def __init__(
        self,
        similarity_threshold: float = 0.08,
        max_sentences_per_chunk: int = 6,
        min_sentences_per_chunk: int = 1,
        similarity_fn: SimilarityFn | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.similarity_fn = similarity_fn or jaccard_similarity

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = split_sentences(document.text)
        if not sentences:
            return []

        groups: list[list[str]] = [[sentences[0]]]
        for sentence in sentences[1:]:
            current_group = groups[-1]
            sim = self.similarity_fn(current_group[-1], sentence)
            should_continue = (
                sim >= self.similarity_threshold
                and len(current_group) < self.max_sentences_per_chunk
            )
            if should_continue:
                current_group.append(sentence)
            else:
                groups.append([sentence])

        # Merge any group below the minimum size into the previous group
        # (avoids single-sentence orphan chunks from an unlucky similarity dip).
        merged: list[list[str]] = []
        for group in groups:
            if merged and len(group) < self.min_sentences_per_chunk:
                merged[-1].extend(group)
            else:
                merged.append(group)

        texts = [" ".join(g) for g in merged]
        return self._build_chunks(document, texts)
