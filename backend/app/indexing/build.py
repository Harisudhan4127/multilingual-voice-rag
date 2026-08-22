"""
Index building core (Phase 3).

One implementation, two entry points:
  - scripts/build_index.py (explicit CLI rebuild)
  - app lifespan AUTO_INDEX_ON_STARTUP (offline-first convenience: if no
    index exists yet, a small dataset is indexed at boot so the demo works
    out of the box)

Steps: load dataset -> chunk with the configured strategy -> embed all
chunks once -> upsert into Qdrant -> persist chunks to JSONL. BM25 is not
persisted; it is rebuilt from the same JSONL in milliseconds.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.chunking.factory import get_chunker
from app.config import Settings
from app.datasets.factory import get_dataset_loader
from app.indexing.base import VectorStore
from app.indexing.embeddings import BaseEmbedder
from app.schemas.models import Chunk

logger = logging.getLogger("voice_rag")


@dataclass
class IndexBuildReport:
    num_documents: int
    num_chunks_total: int
    num_chunks_indexed: int
    dim: int
    vector_mode: str
    duration_ms: float


def retrievable(chunks: list[Chunk]) -> list[Chunk]:
    """Retrieval units exclude parent chunks (they exist for context expansion)."""
    return [c for c in chunks if not c.is_parent]


def chunk_documents(documents: list, strategy: str) -> list[Chunk]:
    """Chunk every document with the named strategy."""
    chunker = get_chunker(strategy)
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))
    return chunks


def save_chunks(path: str | Path, chunks: list[Chunk]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    """Load persisted chunks; raises FileNotFoundError if never built."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Processed chunks not found at {in_path}. "
            "Run `python scripts/build_index.py` first."
        )
    return [Chunk.model_validate(json.loads(line)) for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def build_index(
    settings: Settings,
    embedder: BaseEmbedder,
    vector_store: VectorStore,
    documents: list | None = None,
    strategy: str | None = None,
    chunks_file: str | None = None,
) -> IndexBuildReport:
    start = time.perf_counter()
    strategy = strategy or settings.CHUNKING_STRATEGY
    chunks_file = chunks_file or settings.PROCESSED_CHUNKS_FILE

    if documents is None:
        documents = get_dataset_loader(settings).load_all()

    chunks = chunk_documents(documents, strategy)
    index_targets = retrievable(chunks)
    if not index_targets:
        raise RuntimeError("No chunks produced -- refusing to build an empty index")

    texts = [c.text for c in index_targets]
    t0 = time.perf_counter()
    vectors = embedder.embed(texts)
    embed_ms = (time.perf_counter() - t0) * 1000

    await vector_store.ensure_collection(int(vectors.shape[1]))
    stored = await vector_store.upsert_chunks(index_targets, vectors)

    save_chunks(chunks_file, chunks)
    duration_ms = (time.perf_counter() - start) * 1000

    report = IndexBuildReport(
        num_documents=len(documents),
        num_chunks_total=len(chunks),
        num_chunks_indexed=stored,
        dim=int(vectors.shape[1]),
        vector_mode=vector_store.mode,
        duration_ms=round(duration_ms, 2),
    )
    logger.info(
        "Index built: %d docs -> %d chunks (%d indexed, dim=%d, mode=%s) "
        "in %.1f ms (embeddings %.1f ms)",
        report.num_documents,
        report.num_chunks_total,
        report.num_chunks_indexed,
        report.dim,
        report.vector_mode,
        duration_ms,
        embed_ms,
    )
    return report
