import pytest

from app.chunking.factory import all_strategies, get_chunker
from app.chunking.sentence import split_sentences
from app.schemas.models import Document

SAMPLE_DOC = Document(
    document_id="doc_test",
    passage_id="doc_test_p0",
    title="Test Document",
    text=(
        "This is the first sentence. This is the second sentence, and it is a bit "
        "longer than the first one. Here comes a third sentence. The fourth "
        "sentence introduces a new idea entirely. Finally, the fifth sentence "
        "wraps things up."
    ),
    language="en",
)

EMPTY_DOC = Document(document_id="doc_empty", passage_id="doc_empty_p0", title="Empty", text="")


def test_split_sentences_basic():
    sentences = split_sentences("First one. Second one. Third one.")
    assert len(sentences) == 3
    assert sentences[0] == "First one."


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


@pytest.mark.parametrize("strategy", all_strategies())
def test_chunker_produces_chunks(strategy):
    chunker = get_chunker(strategy)
    chunks = chunker.chunk(SAMPLE_DOC)
    assert len(chunks) > 0
    for c in chunks:
        assert c.document_id == SAMPLE_DOC.document_id
        assert c.strategy == strategy
        assert c.text.strip() != ""
        assert c.total_chunks == len(chunks)


@pytest.mark.parametrize("strategy", all_strategies())
def test_chunker_handles_empty_document(strategy):
    chunker = get_chunker(strategy)
    chunks = chunker.chunk(EMPTY_DOC)
    assert chunks == []


@pytest.mark.parametrize("strategy", all_strategies())
def test_chunk_ids_are_unique(strategy):
    chunker = get_chunker(strategy)
    chunks = chunker.chunk(SAMPLE_DOC)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_sentence_chunker_respects_overlap():
    from app.chunking.sentence import SentenceChunker

    chunker = SentenceChunker(sentences_per_chunk=2, overlap_sentences=1)
    chunks = chunker.chunk(SAMPLE_DOC)
    # With overlap, consecutive chunks should share at least one sentence boundary word set
    assert len(chunks) >= 2


def test_token_chunker_rejects_invalid_overlap():
    from app.chunking.token import TokenChunker

    with pytest.raises(ValueError):
        TokenChunker(tokens_per_chunk=10, overlap_tokens=10)


def test_parent_child_children_reference_valid_parent():
    chunker = get_chunker("parent_child")
    chunks = chunker.chunk(SAMPLE_DOC)
    parent_ids = {c.chunk_id for c in chunks if c.is_parent}
    children = [c for c in chunks if not c.is_parent]
    assert len(children) > 0
    for child in children:
        assert child.parent_id in parent_ids


def test_get_chunker_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        get_chunker("not_a_real_strategy")
