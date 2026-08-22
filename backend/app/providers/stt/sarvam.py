"""
Sarvam AI speech-to-text adapter (Phase 10 / Section 15).

Clean provider isolation: everything Sarvam-specific (endpoint, auth header,
multipart field names, response shape) lives here and nowhere else.

Credentials come exclusively from settings (SARVAM_API_KEY) -- never
hardcoded. Network failures surface as STTError so the API layer can return
controlled responses instead of stack traces.
"""
from __future__ import annotations

import logging

import httpx

from app.config import Settings
from app.providers.stt.base import STTError, STTProvider, Transcript

logger = logging.getLogger("voice_rag")


class SarvamSTTProvider(STTProvider):
    name = "sarvam"

    def __init__(self, settings: Settings) -> None:
        if not settings.SARVAM_API_KEY:
            raise STTError(
                "SARVAM_API_KEY is empty; set it in .env to use STT_PROVIDER=sarvam"
            )
        self._url = settings.SARVAM_BASE_URL.rstrip("/") + settings.SARVAM_STT_PATH
        self._api_key = settings.SARVAM_API_KEY
        self._model = settings.SARVAM_STT_MODEL
        self._timeout_s = settings.STT_TIMEOUT_S

    async def transcribe(
        self, audio: bytes, language: str | None = None
    ) -> Transcript:
        headers = {"api-subscription-key": self._api_key}
        files = {"file": ("audio.webm", audio)}
        data: dict[str, str] = {"model": self._model}
        if language:
            data["language_code"] = language

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(
                    self._url, headers=headers, files=files, data=data
                )
        except httpx.HTTPError as exc:
            logger.error("Sarvam STT request failed: %s", exc)
            raise STTError(f"sarvam request failed: {exc}") from exc

        if response.status_code != 200:
            logger.error(
                "Sarvam STT returned %d: %s", response.status_code, response.text[:200]
            )
            raise STTError(f"sarvam returned HTTP {response.status_code}")

        payload = response.json()
        text = payload.get("transcript")
        if not text:
            raise STTError(f"sarvam response missing transcript: {payload}")
        return Transcript(
            text=text,
            language=language or payload.get("language_code", "unknown"),
            duration_s=payload.get("audio_duration"),
            provider=self.name,
        )
