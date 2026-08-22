"""
Factory that returns the configured STTProvider.

STT_PROVIDER=mock   -> MockSTTProvider (default, offline)
STT_PROVIDER=sarvam -> SarvamSTTProvider (Sarvam AI speech-to-text)

This is the only place that branches on STT_PROVIDER.
"""
from app.config import Settings
from app.providers.stt.base import STTProvider


def get_stt_provider(settings: Settings) -> STTProvider:
    if settings.STT_PROVIDER == "mock":
        from app.providers.stt.mock import MockSTTProvider

        return MockSTTProvider()
    if settings.STT_PROVIDER == "sarvam":
        from app.providers.stt.sarvam import SarvamSTTProvider

        return SarvamSTTProvider(settings)
    raise ValueError(f"Unknown STT_PROVIDER: {settings.STT_PROVIDER}")
