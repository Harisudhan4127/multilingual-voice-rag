"""
Phase 10 tests: Sarvam STT, Real LLM, MSMARCO-XI adapters -- all against
mocked HTTP / in-memory rows. No external services are contacted.
"""
import httpx
import pytest

from app.config import Settings
from app.datasets.msmarco_xi import MSMarcoXIDatasetLoader
from app.providers.llm.factory import get_llm_provider
from app.providers.llm.real import RealLLMProvider
from app.providers.stt.base import STTError
from app.providers.stt.factory import get_stt_provider
from app.providers.stt.sarvam import SarvamSTTProvider


# --- MSMARCO-XI row mapping ---------------------------------------------------

SAMPLE_ROW = {
    "query_id": 118599,
    "query": "supervised learning क्या है?",
    "Eng_Query": "What is supervised learning?",
    "query_type": "DESCRIPTION",
    "passages": {
        "is_selected": [1, 0, 1],
        "English_passages": [
            "Supervised learning is a ML paradigm.",
            "The Taj Mahal is in Agra.",
            "Regression is a supervised task.",
        ],
        "Translated_passages": [
            "पर्यवेक्षित अधिगम एक एमएल प्रतिमान है।",
            "ताजमहल आगरा में है।",
            "प्रतिगमन एक पर्यवेक्षित कार्य है।",
        ],
    },
}


def test_msmarco_row_mapping_translated():
    docs = MSMarcoXIDatasetLoader._row_to_documents(SAMPLE_ROW, "hi", False)
    assert len(docs) == 3
    # Selected passages sorted first.
    assert docs[0].metadata["is_selected"] is True
    assert docs[0].text == SAMPLE_ROW["passages"]["Translated_passages"][0]
    assert docs[0].language == "hi"
    assert docs[0].document_id == "msmarco_118599_0"
    assert docs[0].title == "What is supervised learning?"


def test_msmarco_row_mapping_english_fallback_and_pref():
    row = dict(SAMPLE_ROW)
    docs = MSMarcoXIDatasetLoader._row_to_documents(row, "hi", True)
    assert docs[0].text == SAMPLE_ROW["passages"]["English_passages"][0]

    empty = {"query_id": 1, "passages": {"is_selected": [], "English_passages": [], "Translated_passages": []}}
    assert MSMarcoXIDatasetLoader._row_to_documents(empty, "hi", False) == []


def test_dataset_factory_branch():
    settings = Settings(DATASET_PROVIDER="msmarco_xi", _env_file=None)
    from app.datasets.factory import get_dataset_loader

    loader = get_dataset_loader(settings)
    assert isinstance(loader, MSMarcoXIDatasetLoader)


# --- Sarvam STT ------------------------------------------------------------------


def make_sarvam(**overrides) -> SarvamSTTProvider:
    defaults = dict(
        SARVAM_API_KEY="test-key",
        SARVAM_BASE_URL="https://sarvam.test",
        SARVAM_STT_PATH="/speech-to-text",
        SARVAM_STT_MODEL="saarika:v2",
        STT_TIMEOUT_S=2.0,
        _env_file=None,
    )
    return SarvamSTTProvider(Settings(**{**defaults, **overrides}))


def test_sarvam_requires_api_key():
    with pytest.raises(STTError, match="SARVAM_API_KEY"):
        make_sarvam(SARVAM_API_KEY="")


async def test_sarvam_happy_path(monkeypatch):
    provider = make_sarvam()

    async def fake_post(self, url, **kwargs):
        request = httpx.Request("POST", url, files=kwargs.get("files"))
        return httpx.Response(
            200,
            json={"transcript": "अध्ययन क्या है"},
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    transcript = await provider.transcribe(b"audio-bytes")
    assert transcript.text == "अध्ययन क्या है"
    assert transcript.provider == "sarvam"


async def test_sarvam_http_error(monkeypatch):
    provider = make_sarvam()

    async def fake_post(self, url, **kwargs):
        request = httpx.Request("POST", url)
        return httpx.Response(401, json={"error": "bad key"}, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(STTError, match="401"):
        await provider.transcribe(b"audio")


async def test_sarvam_network_error(monkeypatch):
    provider = make_sarvam()

    async def fail(self, url, **kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx.AsyncClient, "post", fail)
    with pytest.raises(STTError, match="failed"):
        await provider.transcribe(b"audio")


def test_stt_factory_sarvam_branch():
    settings = Settings(STT_PROVIDER="sarvam", SARVAM_API_KEY="k", _env_file=None)
    assert get_stt_provider(settings).name == "sarvam"


# --- Real LLM ----------------------------------------------------------------------


def make_llm(**overrides) -> RealLLMProvider:
    defaults = dict(
        LLM_PROVIDER="real",
        LLM_BASE_URL="https://llm.test/v1",
        LLM_MODEL="test-model",
        LLM_API_KEY="sk-test",
        LLM_TIMEOUT_S=2.0,
        _env_file=None,
    )
    return RealLLMProvider(Settings(**{**defaults, **overrides}))


def test_real_llm_requires_config():
    with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
        make_llm(LLM_BASE_URL="", LLM_MODEL="")


async def test_real_llm_extracts_content(monkeypatch):
    provider = make_llm()
    captured: dict = {}

    async def fake_post(self, url, json=None, headers=None):
        captured["url"] = url
        captured["json"] = json
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"answer": "ok", "citations": [], "confidence": 0.9}'}}
                ]
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    raw = await provider.generate("Question: hi\nJSON response:")
    assert '"answer": "ok"' in raw
    assert captured["json"]["model"] == "test-model"
    assert captured["json"]["temperature"] == 0.0
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["json"]["messages"][-1]["role"] == "user"


async def test_real_llm_http_error(monkeypatch):
    provider = make_llm()

    async def fake_post(self, url, **kwargs):
        request = httpx.Request("POST", url)
        return httpx.Response(500, text="boom", request=request)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        await provider.generate("p")


async def test_real_llm_malformed_envelope(monkeypatch):
    provider = make_llm()

    async def fake_post(self, url, **kwargs):
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"unexpected": {}}, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(RuntimeError, match="envelope"):
        await provider.generate("p")


def test_llm_factory_real_branch():
    settings = Settings(
        LLM_PROVIDER="real",
        LLM_BASE_URL="http://x/v1",
        LLM_MODEL="m",
        _env_file=None,
    )
    assert get_llm_provider(settings).name == "real"
