from app.indexing.bm25_store import BM25Store, tokenize
from app.schemas.models import Chunk


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        document_id="d1",
        chunk_id=chunk_id,
        strategy="sentence",
        position=0,
        total_chunks=1,
        language="en",
        title="t",
        text=text,
    )


def test_tokenize_lowercases_and_strips_punctuation():
    tokens = tokenize("Hello, World! This is BM25.")
    assert tokens == ["hello", "world", "this", "is", "bm25"]


def test_bm25_search_ranks_relevant_chunk_first():
    chunks = [
        make_chunk("c1", "The cat sat on the mat."),
        make_chunk("c2", "Machine learning models require large datasets."),
        make_chunk("c3", "Deep learning is a subset of machine learning."),
    ]
    store = BM25Store()
    store.build(chunks)

    results = store.search("machine learning datasets", top_k=3)
    assert len(results) > 0
    top_chunk, top_score = results[0]
    assert top_chunk.chunk_id in ("c2", "c3")
    assert top_score > 0


def test_bm25_empty_index_returns_empty():
    store = BM25Store()
    assert store.search("anything") == []
    assert len(store) == 0


def test_bm25_empty_query_returns_empty():
    store = BM25Store()
    store.build([make_chunk("c1", "some text here")])
    assert store.search("") == []
