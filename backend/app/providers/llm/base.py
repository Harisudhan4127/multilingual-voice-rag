"""
LLM provider interface (Section 16).

Implementations:
  - MockLLMProvider: deterministic extractive answers from supplied context;
    the whole system is testable offline.
  - RealLLMProvider (Phase 10): OpenAI-compatible HTTP chat API.

Providers return the RAW model output (string). Parsing into structured data
is the generator's job -- providers must not be trusted blindly either.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Return the raw completion for the given prompt."""
        raise NotImplementedError
