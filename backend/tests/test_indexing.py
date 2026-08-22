"""Phase 3 tests: embeddings, vector store, chunk persistence."""
import numpy as np
import pytest

from app.config import Settings
from app.indexing.build import (
    chunk_documents,
    load_chunks,
    retrievable,
    save_chunks,
)
from app.indexing.embeddings import HashingEmbedder
from app.indexing.qdrant_store import QdrantVectorStore
from app.datasets.synthetic import SyntheticDatasetLoader


@pytest.fixture(scope="module")
def documents():
    return SyntheticDatasetLoader().load_all()


@pytest.fixture()
def hashing_embedder():
    return HashingEmbedder(dim=64)


@pytest.fixture()
async def store(tmp_path, settings_local_qdrant):
    store = QdrantVectorStore(settings_local_qdrant())
    yield store
    await store.close()


@pytest.fixture()
def settings_local_qdrant(tmp_path):
    def _make() -> Settings:
        return Settings(
            QDRANT_URL="http://localhost:1",  # nothing listens here
            QDRANT_LOCAL_PATH=str(tmp_path / "qdrant"),
            QDRANT_COLLECTION="test_chunks",
            PROCESSED_CHUNKS_FILE=str(tmp_path / "chunks.jsonl"),
            _env_file=None,
        )

    return _make


def test_hashing_embedder_deterministic_and_normalized(hashing_embedder):
    a = hashing_embedder.embed(["machine learning models", "machine learning models"])
    b = hashing_embedder.embed_one("different text entirely")
    assert a.shape == (2, 64)
    assert np.allclose(a[0], a[1])
    for vec in [*a, b]:
        assert pytest.approx(float(np.linalg.norm(vec)), abs=1e-5) == 1.0


def test_hashing_embedder_similarity_orders_related_text_higher(hashing_embedder):
    query = hashing_embedder.embed_one("gradient descent optimization")
    related = hashing_embedder.embed_one("gradient descent optimizes neural networks")
    unrelated = hashing_embedder.embed_one("the Taj Mahal was built by Shah Jahan")
    assert hashing_embedder.similarity(query, related) > hashing_embedder.similarity(
        query, unrelated
    )


def test_chunk_documents_produces_metadata(documents):
    chunks = chunk_documents(documents[:3], "sentence")
    assert len(chunks) == len(retrievable(chunks))
    chunk = chunks[0]
    assert chunk.document_id == "doc_0001"
    assert chunk.strategy == "sentence"
    assert chunk.position >= 0 and chunk.total_chunks > 0


async def test_qdrant_roundtrip(store, hashing_embedder, documents, tmp_path):
    assert await store.health_check() == "ok"
    assert store.mode == "local"

    chunks = retrievable(chunk_documents(documents[:3], "token"))
    vectors = hashing_embedder.embed([c.text for c in chunks])

    await store.ensure_collection(hashing_embedder.dim)
    stored = await store.upsert_chunks(chunks, vectors)
    assert stored == len(chunks)
    assert await store.count_points() == len(chunks)

    query_vec = hashing_embedder.embed_one(chunks[0].text)
    hits = await store.search(query_vec, top_k=2)
    assert len(hits) <= 2 and hits
    best_chunk, best_score = hits[0]
    # Exact-text self-match must win with cosine similarity ~1.0
    assert best_chunk.chunk_id == chunks[0].chunk_id
    assert best_score > 0.999

    # Idempotent upsert (deterministic point ids)
    await store.upsert_chunks(chunks, vectors)
    assert await store.count_points() == len(chunks)


async def test_ensure_collection_recreates_on_dim_change(
    store, hashing_embedder, documents
):
    chunks = retrievable(chunk_documents(documents[:3], "token"))
    vectors = hashing_embedder.embed([c.text for c in chunks])
    await store.ensure_collection(64)
    await store.upsert_chunks(chunks, vectors)

    other = HashingEmbedder(dim=32)
    await store.ensure_collection(32)  # dim changed -> recreate
    assert await store.count_points() == 0

    await store.ensure_collection(32)  # same dim -> keep data
    await store.upsert_chunks(chunks, other.embed([c.text for c in chunks]))
    assert await store.count_points() == len(chunks)


def test_chunks_jsonl_roundtrip(tmp_path, documents):
    path = tmp_path / "chunks.jsonl"
    chunks = chunk_documents(documents[:2], "sentence")
    save_chunks(path, chunks)
    loaded = load_chunks(path)
    assert loaded == chunks
