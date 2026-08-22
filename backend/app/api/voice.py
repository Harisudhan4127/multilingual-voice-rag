"""
Voice endpoint (Phase 8): audio in -> transcript -> full RAG pipeline.

Multipart form:
    audio           (file, required) the recorded clip
    mock_transcript (text, optional) ONLY honored when STT_PROVIDER=mock --
                    lets demos/tests drive arbitrary questions through the
                    voice path without a microphone. Ignored entirely for
                    real providers so it can never bypass real STT.
"""
import logging

from fastapi import APIRouter, Form, Request, UploadFile

from app.pipeline.orchestrator import PipelineOrchestrator
from app.providers.stt.base import STTError, Transcript
from app.schemas.models import QueryResponse

logger = logging.getLogger("voice_rag")

router = APIRouter(tags=["voice"])

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB cap


@router.post("/voice", response_model=QueryResponse)
async def voice(
    request: Request,
    audio: UploadFile | None = None,
    mock_transcript: str | None = Form(default=None),
) -> QueryResponse:
    stt = getattr(request.app.state, "stt_provider", None)
    orchestrator: PipelineOrchestrator | None = getattr(
        request.app.state, "orchestrator", None
    )
    if stt is None or orchestrator is None:
        return QueryResponse(
            request_id="unavailable",
            answer="Service is starting up; pipeline is not ready yet.",
            confidence=0.0,
            grounded=False,
            latency_ms=0.0,
            status="not_ready",
            error="pipeline not initialized",
        )

    # Demo/test affordance: explicit transcript bypasses only the MOCK stt.
    if stt.name == "mock" and mock_transcript and mock_transcript.strip():
        transcript = Transcript(
            text=mock_transcript.strip(), language="en", provider="mock"
        )
    else:
        if audio is None:
            return _voice_error("no_audio", "No audio file received.")
        raw = await audio.read()
        if not raw:
            return _voice_error("empty_audio", "Uploaded audio is empty.")
        if len(raw) > MAX_AUDIO_BYTES:
            return _voice_error(
                "audio_too_large", f"Audio exceeds {MAX_AUDIO_BYTES} bytes."
            )
        try:
            transcript = await stt.transcribe(raw)
        except STTError as exc:
            logger.warning("STT failed: %s", exc)
            return _voice_error("stt_failed", str(exc))
        except Exception:  # noqa: BLE001 - controlled surface to client
            logger.exception("Unexpected STT failure")
            return _voice_error("stt_failed", "transcription failed")

    return await orchestrator.run(transcript.text, transcript=transcript.text)


def _voice_error(status: str, message: str) -> QueryResponse:
    return QueryResponse(
        request_id="voice-error",
        answer=message,
        confidence=0.0,
        grounded=False,
        latency_ms=0.0,
        status=status,
        error=message,
    )
