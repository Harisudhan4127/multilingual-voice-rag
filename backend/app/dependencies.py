"""
Dependency wiring (Phase 7-8): builds every heavy pipeline component exactly
once and hands them to whoever needs them -- FastAPI's lifespan, benchmark
scripts, evaluation scripts. Nothing else should assemble components by hand,
so there is exactly one source of truth for what the pipeline contains.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings
from app.indexing.bm25_store import BM25Store, tokenize
from app.indexing.build import load_chunks, retrievable
from app.indexing.embeddings import BaseEmbedder, get_embedder
from app.indexing.qdrant_store import QdrantVectorStore
from app.pipeline.generator import RAGGenerator
from app.pipeline.guardrails import Guardrails
from app.pipeline.hybrid_search import HybridRetriever
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.reranker import BaseReranker, get_reranker
from app.providers.llm.base import LLMProvider
from app.providers.llm.factory import get_llm_provider
from app.providers.stt.base import STTProvider
from app.providers.stt.factory import get_stt_provider
from app.utils.metrics import Metrics

logger = logging.getLogger("voice_rag")


@dataclass
class PipelineComponents:
    settings: Settings
    embedder: BaseEmbedder
    vector_store: QdrantVectorStore
    bm25_store: BM25Store
    reranker: BaseReranker
    llm_provider: LLMProvider
    stt_provider: STTProvider
    generator: RAGGenerator
    guardrails: Guardrails
    orchestrator: PipelineOrchestrator

    async def close(self) -> None:
        await self.vector_store.close()


async def ensure_index(
    settings: Settings, embedder: BaseEmbedder, vector_store: QdrantVectorStore
) -> list:
    """Ensure chunks + vector index exist; auto-build if allowed and missing."""
    chunks: list = []
    points = 0
    try:
        chunks = load_chunks(settings.PROCESSED_CHUNKS_FILE)
        points = await vector_store.count_points()
    except FileNotFoundError:
        logger.warning("No processed chunks found at %s", settings.PROCESSED_CHUNKS_FILE)

    if settings.AUTO_INDEX_ON_STARTUP and (not chunks or points == 0):
        logger.info("Auto-building index at startup (%d existing points)", points)
        from app.indexing.build import build_index

        await build_index(settings, embedder, vector_store)
        chunks = load_chunks(settings.PROCESSED_CHUNKS_FILE)
    return chunks


async def build_pipeline(settings: Settings) -> PipelineComponents:
    """Load models/stores once, bootstrap the index, wire the orchestrator."""
    embedder = get_embedder(settings)
    vector_store = QdrantVectorStore(settings)
    chunks = await ensure_index(settings, embedder, vector_store)
    targets = retrievable(chunks)

    bm25_store = BM25Store()
    bm25_store.build(targets)

    reranker = get_reranker(settings)
    llm_provider = get_llm_provider(settings)
    stt_provider = get_stt_provider(settings)

    corpus_vocab: set[str] = set()
    for chunk in targets:
        corpus_vocab.update(tokenize(chunk.text))

    generator = RAGGenerator(llm_provider)
    guardrails = Guardrails(settings, corpus_vocab)
    orchestrator = PipelineOrchestrator(
        settings=settings,
        retriever=HybridRetriever(settings, embedder, vector_store, bm25_store),
        reranker=reranker,
        generator=generator,
        guardrails=guardrails,
        chunks_by_id={c.chunk_id: c for c in chunks},
        metrics=Metrics(),
    )

    return PipelineComponents(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        bm25_store=bm25_store,
        reranker=reranker,
        llm_provider=llm_provider,
        stt_provider=stt_provider,
        generator=generator,
        guardrails=guardrails,
        orchestrator=orchestrator,
    )
