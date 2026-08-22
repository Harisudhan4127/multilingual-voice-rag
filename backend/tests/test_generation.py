"""Phase 5 tests: mock LLM, prompt building, structured output, retries."""
import json

import pytest
from pydantic import ValidationError

from app.datasets.synthetic import SyntheticDatasetLoader
from app.indexing.build import chunk_documents
from app.pipeline.generator import (
    GeneratedAnswer,
    GenerationError,
    RAGGenerator,
    build_prompt,
    select_context,
)
from app.providers.llm.base import LLMProvider
from app.providers.llm.factory import get_llm_provider
from app.providers.llm.mock import MockLLMProvider


@pytest.fixture(scope="module")
def chunks():
    return chunk_documents(SyntheticDatasetLoader().load_all()[:2], "sentence")


async def test_mock_llm_answers_from_context(chunks):
    provider = MockLLMProvider()
    query = "What is supervised learning?"
    relevant = [c for c in chunks if "supervised learning" in c.text.lower()]
    assert relevant  # fixture sanity

    raw = await provider.generate(build_prompt(query, relevant[:3]))
    data = json.loads(raw)  # strict JSON out of the mock too
    assert "supervised" in data["answer"].lower()
    assert data["citations"]
    assert all(cid in {c.chunk_id for c in relevant} for cid in data["citations"])
    assert 0.0 <= data["confidence"] <= 1.0


async def test_mock_llm_refuses_without_overlap(chunks):
    provider = MockLLMProvider()
    unrelated = [c for c in chunks if "supervised" not in c.text.lower()]
    raw = await provider.generate(
        build_prompt("quantum chromodynamics lagrangian", unrelated)
    )
    data = json.loads(raw)
    assert "enough information" in data["answer"]
    assert data["citations"] == []
    assert data["confidence"] < 0.1


async def test_generator_validates_and_cites(chunks):
    generator = RAGGenerator(MockLLMProvider())
    query = "What is supervised learning?"
    context = [c for c in chunks if "supervised" in c.text.lower()][:3]
    answer = await generator.generate(query, context)
    assert isinstance(answer, GeneratedAnswer)
    assert answer.answer
    valid_ids = {c.chunk_id for c in context}
    assert set(answer.citations) <= valid_ids


class BrokenProvider(LLMProvider):
    """Returns garbage first (n=0), then a fixed payload."""

    def __init__(self, payloads: list[str]):
        self.payloads = payloads
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        raw = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return raw


async def test_generator_retries_once_then_succeeds():
    bad = "this is definitely not json"
    good = json.dumps({"answer": "fine", "citations": [], "confidence": 0.5})
    provider = BrokenProvider([bad, good])
    answer = await RAGGenerator(provider).generate("q", [])
    assert answer.answer == "fine"
    assert provider.calls == 2


async def test_generator_fails_controlled_after_retries():
    provider = BrokenProvider(["nope"])
    with pytest.raises(GenerationError, match="invalid LLM response"):
        await RAGGenerator(provider).generate("q", [])
    assert provider.calls == 2  # initial attempt + exactly one retry


async def test_generator_tolerates_markdown_fences():
    fenced = '```json\n{"answer": "ok", "citations": ["c1"], "confidence": 0.9}\n```'
    answer = await RAGGenerator(BrokenProvider([fenced])).generate("q", [])
    assert answer.answer == "ok"
    assert answer.citations == ["c1"]


def test_generated_answer_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        GeneratedAnswer(answer="x", confidence=1.5)


def test_select_context_expands_parent():
    child = chunk_documents(SyntheticDatasetLoader().load_all()[:1], "parent_child")
    parents = [c for c in child if c.is_parent]
    kids = [c for c in child if not c.is_parent]
    by_id = {c.chunk_id: c for c in child}

    selected = select_context([(kids[0], 1.0)], by_id)
    assert selected[0].chunk_id == kids[0].parent_id
    assert selected[0] in parents


def test_select_context_passthrough_without_parents(chunks):
    by_id = {c.chunk_id: c for c in chunks}
    selected = select_context([(chunks[0], 1.0)], by_id)
    assert selected[0].chunk_id == chunks[0].chunk_id


def test_get_llm_provider_factory():
    from app.config import Settings

    provider = get_llm_provider(Settings(LLM_PROVIDER="mock", _env_file=None))
    assert provider.name == "mock"
    # 'real' branch now constructs the adapter; it requires endpoint config.
    with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
        get_llm_provider(
            Settings(LLM_PROVIDER="real", LLM_BASE_URL="", LLM_MODEL="", _env_file=None)
        )
