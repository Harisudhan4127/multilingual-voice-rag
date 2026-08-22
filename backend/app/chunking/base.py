"""
Common Chunker interface. Every strategy implements `chunk(document)` and
returns a list of Chunk objects with consistent metadata, so the rest of the
pipeline (indexing, retrieval) never needs to know which strategy produced
a given chunk.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.models import Chunk, Document


class Chunker(ABC):
    strategy_name: str = "base"

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError

    def _build_chunks(
        self,
        document: Document,
        texts: list[str],
        parent_id: str | None = None,
        is_parent: bool = False,
    ) -> list[Chunk]:
        """Shared helper: wraps raw text pieces into Chunk objects with metadata."""
        total = len(texts)
        chunks = []
        for i, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue
            suffix = "parent" if is_parent else "child"
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    chunk_id=f"{document.document_id}_{self.strategy_name}_{suffix}_{i}",
                    strategy=self.strategy_name,
                    position=i,
                    total_chunks=total,
                    language=document.language,
                    title=document.title,
                    text=text,
                    parent_id=parent_id,
                    is_parent=is_parent,
                )
            )
        return chunks
