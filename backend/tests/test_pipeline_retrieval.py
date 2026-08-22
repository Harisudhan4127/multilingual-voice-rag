"""Phase 4 tests: query processing, retrievers, RRF fusion, reranking."""
import numpy as np
import pytest

from app.config import Settings
from app.datasets.eval_queries import EVAL_QUERIES
from app.datasets.synthetic import SyntheticDatasetLoader
from app.indexing.bm25_store import BM25Store
from app.indexing.build import chunk_documents, retrievable
from app.indexing.embeddings import HashingEmbedder
from app.indexing.qdrant_store import QdrantVectorStore
from app.pipeline.hybrid_search import HybridRetriever, rrf_fuse
from app.pipeline.query_processor import QueryProcessor
from app.pipeline.reranker import LexicalReranker
from app.schemas.models import Chunk


def make_chunk(cid: str, text: str = "text") -> Chunk:
    return Chunk(
        document_id=cid,
        chunk_id=cid,
        strategy="sentence",
        position=0,
        total_chunks=1,
        language="en",
        title="t",
        text=text,
    )


# --- Query processor ------------------------------------------------------


@pytest.fixture()
def processor():
    return QueryProcessor()


def test_query_processor_normalizes_whitespace(processor):
    out = processor.process("  What   is\tsupervised   learning? ")
    assert out.cleaned == "What is supervised learning?"
    assert out.valid


def test_query_processor_strips_speech_artifacts(processor):
    out = processor.process("Um, uh what is machine learning?")
    assert out.cleaned == "what is machine learning?"
    assert out.original.startswith("Um")


def test_query_processor_detects_language(processor):
    assert processor.process("मशीन लर्निंग क्या है").language == "hi"
    assert processor.process("What is ML?").language == "en"


def test_query_processor_classifies_question_vs_keyword(processor):
    assert processor.process("What is overfitting?").query_type == "question"
    assert processor.process("overfitting prevention").query_type == "keyword"


def test_query_processor_rejects_empty(processor):
    for raw in ("", "   ", "um uh er", "?"):
        out = processor.process(raw)
        assert not out.valid
        assert out.invalid_reason


# --- RRF fusion ------------------------------------------------------------


def test_rrf_fuse_interleaves_and_dedups():
    a1, a2, a3 = make_chunk("a1"), make_chunk("a2"), make_chunk("a3")
    dense = [(a1, 0.9), (a2, 0.8)]
    bm25 = [(a3, 12.0), (a2, 9.0)]

    fused = rrf_fuse(dense, bm25, rrf_k=60, dense_weight=0.65, bm25_weight=0.35)

    ids = [c.chunk_id for c, _ in fused]
    assert len(ids) == len(set(ids)) == 3
    # a2 appears in both lists -> highest fused score.
    assert ids[0] == "a2"
    scores = dict(zip(ids, [s for _, s in fused]))
    # weighted RRF math (rank starts at 1): a1 = .65/61,
    # a2 = rank 2 in both lists = .65/62 + .35/62, a3 = .35/61
    assert scores["a1"] == pytest.approx(0.65 / 61)
    assert scores["a2"] == pytest.approx(0.65 / 62 + 0.35 / 62)
    assert scores["a3"] == pytest.approx(0.35 / 61)


def test_rrf_fuse_respects_top_k():
    chunks = [make_chunk(f"c{i}") for i in range(5)]
    dense = [(c, 1.0) for c in chunks]
    fused = rrf_fuse(dense, [], rrf_k=60, dense_weight=1.0, bm25_weight=0.0, top_k=2)
    assert len(fused) == 2


def test_rrf_fuse_weights_matter():
    c1 = make_chunk("only_dense")
    fused = rrf_fuse([(c1, 1.0)], [], rrf_k=60, dense_weight=0.9, bm25_weight=0.1)
    assert fused[0][1] == pytest.approx(0.9 / 61)


# --- Retrievers (hashing embedder + local qdrant + real BM25) --------------


@pytest.fixture(scope="module")
def corpus_chunks():
    documents = SyntheticDatasetLoader().load_all()
    return chunk_documents(documents, "sentence")


@pytest.fixture()
async def hybrid(tmp_path, corpus_chunks):
    settings = Settings(
        QDRANT_URL="http://localhost:1",
        QDRANT_LOCAL_PATH=str(tmp_path / "qdrant"),
        QDRANT_COLLECTION="test_hybrid",
        _env_file=None,
    )
    embedder = HashingEmbedder(dim=64)
    store = QdrantVectorStore(settings)
    await store.ensure_collection(embedder.dim)
    targets = retrievable(corpus_chunks)
    vectors = embedder.embed([c.text for c in targets])
    await store.upsert_chunks(targets, vectors)

    bm25 = BM25Store()
    bm25.build(targets)
    yield HybridRetriever(settings, embedder, store, bm25), embedder
    await store.close()


async def test_bm25_retriever_finds_relevant_doc(hybrid, corpus_chunks):
    _, _ = hybrid
    bm25 = BM25Store()
    bm25.build(retrievable(corpus_chunks))
    results = await BM25RetrieverWrapper(bm25).search(
        EVAL_QUERIES[0].query, top_k=5
    )
    hit_ids = {chunk.document_id for chunk, _ in results}
    assert EVAL_QUERIES[0].relevant_document_ids[0] in hit_ids


class BM25RetrieverWrapper:
    """Tiny async adapter mirroring pipeline.retriever.BM25Retriever."""

    def __init__(self, store):
        self._store = store

    async def search(self, query, top_k):
        return self._store.search(query, top_k)


async def test_dense_retriever_self_match(hybrid):
    hybrid_ret, embedder = hybrid
    from app.pipeline.retriever import DenseRetriever

    target = retrievable(chunk_documents(SyntheticDatasetLoader().load_all()[:1], "sentence"))[0]
    dense = DenseRetriever(embedder, hybrid_ret.dense._vector_store)
    results = await dense.search(target.text[:120], top_k=3)
    assert results[0][0].chunk_id == target.chunk_id


async def test_hybrid_runs_both_legs(hybrid):
    hybrid_ret, _ = hybrid
    results = await hybrid_ret.search(EVAL_QUERIES[4].query, top_k=10)
    assert results
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)  # sorted best-first
    assert all(s > 0 for s in scores)


# --- Reranker ----------------------------------------------------------------


async def test_lexical_reranker_prefers_relevant_chunk():
    reranker = LexicalReranker()
    query = "gradient descent optimizer"
    relevant = (
        make_chunk("rel", "Gradient descent is an optimization algorithm."),
        0.1,
    )
    irrelevant = (make_chunk("irr", "The Taj Mahal is in Agra, India."), 0.9)
    reranked = await reranker.rerank(query, [irrelevant, relevant], top_k=2)
    assert reranked[0][0].chunk_id == "rel"
    assert reranked[0][1] > reranked[1][1]


async def test_lexical_reranker_top_k_and_empty():
    reranker = LexicalReranker()
    assert await reranker.rerank("q", [], top_k=5) == []
    candidates = [(make_chunk(f"c{i}", f"token{i} unique words here"), 1.0) for i in range(6)]
    out = await reranker.rerank("token unique", candidates, top_k=3)
    assert len(out) == 3


def test_hashing_embedder_dim_matches_config():
    e = HashingEmbedder(dim=384)
    assert e.embed_one("x").shape == (384,)
