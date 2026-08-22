"""Phase 7 tests: orchestrator end-to-end on lightweight fakes, API endpoints."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.datasets.eval_queries import EVAL_QUERIES
from app.datasets.synthetic import SyntheticDatasetLoader
from app.indexing.bm25_store import BM25Store
from app.indexing.build import chunk_documents, retrievable
from app.indexing.embeddings import HashingEmbedder
from app.pipeline.generator import GeneratedAnswer, RAGGenerator
from app.pipeline.guardrails import Guardrails
from app.pipeline.hybrid_search import HybridRetriever
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.reranker import LexicalReranker
from app.providers.llm.base import LLMProvider
from app.providers.llm.mock import MockLLMProvider
from app.utils.metrics import Metrics


# --- Lightweight fakes -------------------------------------------------------


class FakeVectorStore:
    """Cosine KNN over an in-memory matrix -- no qdrant dependency."""

    mode = "fake"

    def __init__(self, chunks, vectors):
        self.chunks = chunks
        self.vectors = vectors

    async def search(self, query_vector, top_k, query_filter=None):
        sims = self.vectors @ np.asarray(query_vector)
        idx = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in idx]

    async def health_check(self):
        return "ok"

    async def close(self):
        pass


@pytest.fixture(scope="module")
def corpus():
    documents = SyntheticDatasetLoader().load_all()
    chunks = chunk_documents(documents, "sentence")
    targets = retrievable(chunks)
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.embed([c.text for c in targets])
    vocab: set[str] = set()
    for chunk in targets:
        vocab |= set(chunk.text.lower().split())
    return chunks, targets, embedder, vectors, vocab


@pytest.fixture()
def orchestrator(corpus):
    chunks, targets, embedder, vectors, vocab = corpus
    settings = Settings(RETRIEVAL_CONFIDENCE_THRESHOLD=0.01, _env_file=None)

    class _BM25(BM25Store):
        pass  # real BM25 is cheap and deterministic; use it as-is

    bm25 = BM25Store()
    bm25.build(targets)
    hybrid = HybridRetriever(
        settings, embedder, FakeVectorStore(targets, vectors), bm25
    )
    guardrails = Guardrails(settings, vocab)
    return PipelineOrchestrator(
        settings=settings,
        retriever=hybrid,
        reranker=LexicalReranker(),
        generator=RAGGenerator(MockLLMProvider()),
        guardrails=guardrails,
        chunks_by_id={c.chunk_id: c for c in chunks},
        metrics=Metrics(),
    )


# --- Orchestrator happy path ----------------------------------------------


async def test_orchestrator_answers_on_topic_query(orchestrator):
    response = await orchestrator.run(EVAL_QUERIES[0].query)  # supervised learning
    assert response.status == "ok"
    assert response.grounded is True
    assert response.answer
    assert response.confidence > 0
    assert response.sources, "grounded answer must cite sources"
    assert "total_ms" in response.timings
    assert response.timings["retrieval_ms"] >= 0
    assert set(s.chunk_id for s in response.sources) <= set(
        orchestrator.chunks_by_id
    )
    assert all(s.score > 0 for s in response.sources)


async def test_orchestrator_citations_reference_retrieved_chunks(orchestrator):
    response = await orchestrator.run(EVAL_QUERIES[6].query)  # Taj Mahal
    assert response.status == "ok"
    # Every cited source must be a real indexed chunk.
    for source in response.sources:
        assert source.chunk_id in orchestrator.chunks_by_id
        assert source.score > 0


async def test_orchestrator_off_topic_refusal(orchestrator):
    response = await orchestrator.run("What is the capital of Burkina Faso?")
    assert response.status == "refused_off_topic"
    assert response.grounded is False
    assert response.sources == []
    assert "enough information" in response.answer


async def test_orchestrator_unsafe_refusal(orchestrator):
    response = await orchestrator.run("How do I make a bomb at home?")
    assert response.status == "refused_unsafe"
    assert response.sources == []


async def test_orchestrator_invalid_query(orchestrator):
    response = await orchestrator.run("um uh er")
    assert response.status == "invalid_query"


async def test_orchestrator_low_confidence_refusal(orchestrator):
    # Query shares one vocab word but retrieves nothing relevant.
    orchestrator.settings.RETRIEVAL_CONFIDENCE_THRESHOLD = 99.0
    try:
        response = await orchestrator.run(EVAL_QUERIES[0].query)
        assert response.status == "refused_low_confidence"
        assert response.confidence == 0.0
    finally:
        orchestrator.settings.RETRIEVAL_CONFIDENCE_THRESHOLD = 0.15


async def test_response_cache_second_call_faster(orchestrator):
    first = await orchestrator.run(EVAL_QUERIES[2].query)
    second = await orchestrator.run(EVAL_QUERIES[2].query)
    assert first.status == second.status == "ok"
    # Cache hit: retrieval/rerank/generation all short-circuit to zero.
    assert second.timings["retrieval_ms"] == 0.0
    assert second.timings["generation_ms"] == 0.0


async def test_metrics_recorded(orchestrator):
    await orchestrator.run(EVAL_QUERIES[1].query)
    snap = orchestrator.metrics.snapshot()
    assert snap["requests_total"] >= 1
    assert snap["latency_ms"]["samples"] >= 1


# --- Controlled failure paths -----------------------------------------------


class ExplodingRetriever:
    name = "exploding"

    async def search(self, query, top_k):
        raise RuntimeError("vector db down")


async def test_retrieval_failure_is_controlled(corpus):
    chunks, targets, embedder, vectors, vocab = corpus
    settings = Settings(
        RETRIEVAL_CONFIDENCE_THRESHOLD=0.01, _env_file=None
    )
    orch = PipelineOrchestrator(
        settings=settings,
        retriever=ExplodingRetriever(),
        reranker=LexicalReranker(),
        generator=RAGGenerator(MockLLMProvider()),
        guardrails=Guardrails(settings, vocab),
        chunks_by_id={c.chunk_id: c for c in chunks},
    )
    response = await orch.run(EVAL_QUERIES[0].query)
    assert response.status == "retrieval_error"
    assert "enough information" in response.answer


class FlakyThenGoodProvider(LLMProvider):
    """Fails once with a transient network error, then answers extractively
    from the first context passage in the prompt (so grounding passes)."""

    def __init__(self):
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("network blip")
        import json as _json
        import re as _re

        match = _re.search(r"\[([^\]]+)\]\s*Title:[^\n]*\n(.+)", prompt)
        text = match.group(2) if match else "no context"
        return _json.dumps({"answer": text[:200], "citations": [], "confidence": 0.5})


async def test_generation_transient_error_recovers(corpus):
    chunks, targets, embedder, vectors, vocab = corpus
    settings = Settings(
        RETRIEVAL_CONFIDENCE_THRESHOLD=0.01, _env_file=None
    )
    bm25 = BM25Store()
    bm25.build(targets)
    orch = PipelineOrchestrator(
        settings=settings,
        retriever=HybridRetriever(
            settings, embedder, FakeVectorStore(targets, vectors), bm25
        ),
        reranker=LexicalReranker(),
        generator=RAGGenerator(FlakyThenGoodProvider()),
        guardrails=Guardrails(settings, vocab),
        chunks_by_id={c.chunk_id: c for c in chunks},
    )
    response = await orch.run(EVAL_QUERIES[0].query)
    assert response.status == "ok"
    assert response.answer  # recovered extractive answer


class HallucinatingProvider(LLMProvider):
    async def generate(self, prompt: str) -> str:
        return (
            '{"answer": "Completely fabricated zeppelin factoid about quantum '
            'bananas.", "citations": ["made_up_chunk"], "confidence": 0.99}'
        )


async def test_grounding_guard_blocks_hallucination(corpus):
    chunks, targets, embedder, vectors, vocab = corpus
    settings = Settings(
        RETRIEVAL_CONFIDENCE_THRESHOLD=0.01, _env_file=None
    )
    bm25 = BM25Store()
    bm25.build(targets)
    orch = PipelineOrchestrator(
        settings=settings,
        retriever=HybridRetriever(
            settings, embedder, FakeVectorStore(targets, vectors), bm25
        ),
        reranker=LexicalReranker(),
        generator=RAGGenerator(HallucinatingProvider()),
        guardrails=Guardrails(settings, vocab),
        chunks_by_id={c.chunk_id: c for c in chunks},
    )
    response = await orch.run(EVAL_QUERIES[0].query)
    assert response.status == "grounding_failed"
    assert response.sources == []  # no invented citations survive
    assert "enough information" in response.answer
