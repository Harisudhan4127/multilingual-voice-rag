"""Phase 8 tests: mock STT provider + voice endpoint."""
import pytest

from app.config import Settings
from app.providers.stt.base import STTError
from app.providers.stt.factory import get_stt_provider
from app.providers.stt.mock import MockSTTProvider


# --- MockSTTProvider ---------------------------------------------------------


async def test_mock_stt_rejects_empty_audio():
    with pytest.raises(STTError, match="no audio"):
        await MockSTTProvider().transcribe(b"")


async def test_mock_stt_deterministic_per_audio():
    provider = MockSTTProvider()
    t1 = await provider.transcribe(b"fake-wav-bytes-123")
    t2 = await provider.transcribe(b"fake-wav-bytes-123")
    assert t1.text == t2.text
    assert t1.provider == "mock"
    assert t1.text  # a real question came out


async def test_mock_stt_varies_across_inputs():
    provider = MockSTTProvider()
    texts = {(await provider.transcribe(bytes([i]))).text for i in range(8)}
    assert len(texts) > 1


def test_get_stt_provider_factory():
    provider = get_stt_provider(Settings(STT_PROVIDER="mock", _env_file=None))
    assert provider.name == "mock"
    # 'sarvam' branch now constructs the adapter; it requires an API key.
    with pytest.raises(STTError, match="SARVAM_API_KEY"):
        get_stt_provider(Settings(STT_PROVIDER="sarvam", SARVAM_API_KEY="", _env_file=None))


# --- Voice endpoint ------------------------------------------------------------


def test_voice_with_mock_transcript_runs_pipeline(client):
    res = client.post(
        "/api/v1/voice",
        data={"mock_transcript": "What is supervised learning?"},
        files={"audio": ("clip.webm", b"RIFF....", "audio/webm")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["grounded"] is True
    assert body["sources"]
    # The transcript must have been the override text, not the canned one.
    assert "supervised" in body["answer"].lower()


def test_voice_canned_transcription_without_override(client):
    res = client.post(
        "/api/v1/voice",
        files={"audio": ("clip.webm", b"some-real-audio-bytes", "audio/webm")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in ("ok", "refused_off_topic")  # canned question flows
    assert body["latency_ms"] >= 0


def test_voice_missing_audio(client):
    res = client.post("/api/v1/voice", data={})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "no_audio"


def test_voice_empty_audio_file(client):
    res = client.post(
        "/api/v1/voice",
        files={"audio": ("clip.webm", b"", "audio/webm")},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "empty_audio"


def test_voice_oversized_audio_rejected(client):
    big = b"x" * (10 * 1024 * 1024 + 1)
    res = client.post(
        "/api/v1/voice",
        files={"audio": ("big.webm", big, "audio/webm")},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "audio_too_large"


def test_voice_counts_toward_voice_metrics(client):
    before = client.get("/api/v1/metrics").json()["voice_requests_total"]
    client.post(
        "/api/v1/voice",
        data={"mock_transcript": "Why did Ashoka convert to Buddhism?"},
        files={"audio": ("c.webm", b"abc", "audio/webm")},
    )
    after = client.get("/api/v1/metrics").json()["voice_requests_total"]
    assert after == before + 1
