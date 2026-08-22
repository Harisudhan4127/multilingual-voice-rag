"""
MockSTTProvider (Phase 8): deterministic offline "speech-to-text".

Behaviour:
  - empty audio -> STTError (exercises the no-microphone-input path)
  - explicit override text (dev/demo affordance, see api/voice.py) -> used as-is
  - otherwise -> stable pseudo-transcription derived from the audio bytes,
    chosen from canned MSMARCO-style questions so the whole voice loop can be
    demoed and tested with zero external services.
"""
from __future__ import annotations

import hashlib

from app.providers.stt.base import STTError, STTProvider, Transcript

_CANNED_TRANSCRIPTS: tuple[str, ...] = (
    "What is supervised learning?",
    "How do you prevent overfitting in a machine learning model?",
    "Why did Ashoka convert to Buddhism?",
    "How do greenhouse gases trap heat in the atmosphere?",
    "What is the difference between a git commit and a branch?",
    "What part of the brain controls the sleep-wake cycle?",
)


class MockSTTProvider(STTProvider):
    name = "mock"

    def __init__(self) -> None:
        self.override_text: str | None = None

    async def transcribe(
        self, audio: bytes, language: str | None = None
    ) -> Transcript:
        if not audio or not audio.strip():
            raise STTError("no audio received")

        if self.override_text is not None:
            return Transcript(
                text=self.override_text, language=language or "en", provider=self.name
            )

        digest = hashlib.sha256(audio).digest()
        index = int.from_bytes(digest[:4], "little") % len(_CANNED_TRANSCRIPTS)
        # A rough "duration" estimate purely for display purposes: ~16 kB/s.
        duration = round(len(audio) / 16000, 2)
        return Transcript(
            text=_CANNED_TRANSCRIPTS[index],
            language=language or "en",
            duration_s=duration,
            provider=self.name,
        )
