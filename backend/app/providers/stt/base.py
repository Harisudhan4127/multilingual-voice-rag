"""
STT provider interface (Section 15).

Implementations:
  - MockSTTProvider: offline deterministic transcription for dev/tests.
  - SarvamSTTProvider (Phase 10): real Sarvam AI speech-to-text API.

Credentials never appear in code -- they come from settings only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class Transcript(BaseModel):
    text: str
    language: str = "en"
    duration_s: float | None = None
    provider: str = "unknown"


class STTError(Exception):
    """Raised when transcription fails in a controlled, user-facing way."""


class STTProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def transcribe(
        self, audio: bytes, language: str | None = None
    ) -> Transcript:
        raise NotImplementedError
