"""
Shared lightweight pipeline fixtures for API-level tests.

Deliberately model-free: HashingEmbedder + FakeVectorStore + real BM25 +
MockLLM/MockSTT give fast, deterministic end-to-end HTTP tests without
loading torch models or Qdrant.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.datasets.synthetic import SyntheticDatasetLoader
from app.indexing.bm25_store import BM25Store
from app.indexing.build import chunk_documents, retrievable
from app.indexing.embeddings import HashingEmbedder
from app.main import app
from app.pipeline.generator import RAGGenerator
from app.pipeline.guardrails import Guardrails
from app.pipeline.hybrid_search import HybridRetriever
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.reranker import LexicalReranker
from app.providers.llm.mock import MockLLMProvider
from app.providers.stt.mock import MockSTTProvider
from app.utils.metrics import Metrics


class FakeVectorStore:
    """Cosine KNN over an in-memory matrix."""

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


def build_test_orchestrator(settings: Settings | None = None):
    """Assemble a fully wired orchestrator on synthetic data, no heavy deps."""
    settings = settings or Settings(RETRIEVAL_CONFIDENCE_THRESHOLD=0.01, _env_file=None)
    documents = SyntheticDatasetLoader().load_all()
    chunks = chunk_documents(documents, "sentence")
    targets = retrievable(chunks)
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.embed([c.text for c in targets])

    bm25 = BM25Store()
    bm25.build(targets)
    vocab: set[str] = set()
    for c in targets:
        vocab |= set(c.text.lower().split())

    store = FakeVectorStore(targets, vectors)
    orchestrator = PipelineOrchestrator(
        settings=settings,
        retriever=HybridRetriever(settings, embedder, store, bm25),
        reranker=LexicalReranker(),
        generator=RAGGenerator(MockLLMProvider()),
        guardrails=Guardrails(settings, vocab),
        chunks_by_id={c.chunk_id: c for c in chunks},
        metrics=Metrics(),
    )
    return orchestrator, embedder, store, bm25


@pytest.fixture(scope="module")
def client():
    """
    TestClient WITHOUT a context manager: entering it as a context manager
    would run the app lifespan and load the real embedding/reranker models.
    Components are attached to app.state directly instead.
    """
    orchestrator, embedder, store, bm25 = build_test_orchestrator()
    app.state.orchestrator = orchestrator
    app.state.embedder = embedder
    app.state.vector_store = store
    app.state.bm25_store = bm25
    app.state.reranker = LexicalReranker()
    app.state.stt_provider = MockSTTProvider()

    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        app.state.orchestrator = None
        app.state.embedder = None
        app.state.vector_store = None
        app.state.bm25_store = None
        app.state.reranker = None
        app.state.stt_provider = None
