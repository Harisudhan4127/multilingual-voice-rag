"""
Factory that returns the configured LLMProvider.

LLM_PROVIDER=mock -> MockLLMProvider (default, offline)
LLM_PROVIDER=real -> RealLLMProvider (OpenAI-compatible HTTP API)

This is the only place that branches on LLM_PROVIDER.
"""
from app.config import Settings
from app.providers.llm.base import LLMProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    if settings.LLM_PROVIDER == "mock":
        from app.providers.llm.mock import MockLLMProvider

        return MockLLMProvider()
    if settings.LLM_PROVIDER == "real":
        from app.providers.llm.real import RealLLMProvider

        return RealLLMProvider(settings)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")
