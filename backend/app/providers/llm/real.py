"""
RealLLMProvider (Phase 10): OpenAI-compatible chat-completions over HTTP.

Works with any OpenAI-compatible endpoint (OpenAI, Azure-style gateways,
vLLM/Ollama servers, ...) configured through:
    LLM_BASE_URL   e.g. https://api.openai.com/v1
    LLM_API_KEY    bearer token
    LLM_MODEL      model name

The provider returns the RAW completion string; structured-output validation,
retry-on-invalid and grounding all happen downstream in the generator /
orchestrator exactly as they do for the mock provider.
"""
from __future__ import annotations

import logging

import httpx

from app.config import Settings
from app.pipeline.generator import PROMPT_TEMPLATE  # shared RAG instructions
from app.providers.llm.base import LLMProvider

logger = logging.getLogger("voice_rag")


class RealLLMProvider(LLMProvider):
    name = "real"

    def __init__(self, settings: Settings) -> None:
        if not settings.LLM_BASE_URL or not settings.LLM_MODEL:
            raise RuntimeError(
                "LLM_PROVIDER=real requires LLM_BASE_URL and LLM_MODEL in .env"
            )
        self._url = settings.LLM_BASE_URL.rstrip("/") + "/chat/completions"
        self._api_key = settings.LLM_API_KEY
        self._model = settings.LLM_MODEL
        self._json_mode = settings.LLM_JSON_MODE
        # Slightly above the orchestrator's LLM_TIMEOUT_S so the pipeline's
        # timeout fires first and stays the single source of truth.
        self._timeout_s = settings.LLM_TIMEOUT_S + 5.0

    async def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a grounded question-answering assistant. Follow the "
                    "user's formatting rules exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        body: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
        }
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(self._url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            logger.error("LLM request failed: %s", exc)
            raise RuntimeError(f"llm request failed: {exc}") from exc

        if response.status_code != 200:
            logger.error(
                "LLM returned %d: %s", response.status_code, response.text[:200]
            )
            raise RuntimeError(f"llm returned HTTP {response.status_code}")

        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise RuntimeError(f"malformed LLM response envelope: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM returned empty content")
        return content


# PROMPT_TEMPLATE import guard: the RAG instructions must match what the mock
# provider consumes, so importing keeps them in lockstep.
assert "Answer ONLY using the supplied context" in PROMPT_TEMPLATE
