"""
Parent-child chunking: splits a document into large "parent" chunks (for
full context) and, within each parent, smaller "child" chunks (for precise
retrieval matching). Children carry parent_id so the pipeline can retrieve
on the child's precision and then expand to the parent's context before
sending to the LLM (a standard RAG pattern for balancing retrieval
precision against generation context quality).

Both parent and child chunks are returned; indexing decides whether to
embed/search only children (typical) while keeping parents available for
context expansion.
"""
from __future__ import annotations

from app.chunking.base import Chunker
from app.chunking.sentence import split_sentences
from app.schemas.models import Chunk, Document


class ParentChildChunker(Chunker):
    strategy_name = "parent_child"

    def __init__(
        self,
        parent_sentences: int = 6,
        child_sentences: int = 2,
        child_overlap: int = 0,
    ) -> None:
        if child_sentences > parent_sentences:
            raise ValueError("child_sentences must not exceed parent_sentences")
        self.parent_sentences = parent_sentences
        self.child_sentences = child_sentences
        self.child_overlap = child_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = split_sentences(document.text)
        if not sentences:
            return []

        all_chunks: list[Chunk] = []

        for p_idx in range(0, len(sentences), self.parent_sentences):
            parent_sentence_group = sentences[p_idx : p_idx + self.parent_sentences]
            parent_text = " ".join(parent_sentence_group)
            parent_id = (
                f"{document.document_id}_{self.strategy_name}_parent_"
                f"{p_idx // self.parent_sentences}"
            )

            all_chunks.append(
                Chunk(
                    document_id=document.document_id,
                    chunk_id=parent_id,
                    strategy=self.strategy_name,
                    position=p_idx // self.parent_sentences,
                    total_chunks=0,  # filled in below once total is known
                    language=document.language,
                    title=document.title,
                    text=parent_text,
                    parent_id=None,
                    is_parent=True,
                )
            )

            step = max(self.child_sentences - self.child_overlap, 1)
            c_idx = 0
            child_pos = 0
            while c_idx < len(parent_sentence_group):
                child_group = parent_sentence_group[c_idx : c_idx + self.child_sentences]
                child_text = " ".join(child_group)
                if child_text.strip():
                    all_chunks.append(
                        Chunk(
                            document_id=document.document_id,
                            chunk_id=f"{parent_id}_child_{child_pos}",
                            strategy=self.strategy_name,
                            position=child_pos,
                            total_chunks=0,
                            language=document.language,
                            title=document.title,
                            text=child_text,
                            parent_id=parent_id,
                            is_parent=False,
                        )
                    )
                    child_pos += 1
                if c_idx + self.child_sentences >= len(parent_sentence_group):
                    break
                c_idx += step

        total = len(all_chunks)
        for c in all_chunks:
            c.total_chunks = total

        return all_chunks
