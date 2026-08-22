"""
RAG generation stage (Phase 5 / Sections 17-18).

Responsibilities:
  - build the grounded RAG prompt (answer ONLY from context, cite chunks),
  - call the configured LLMProvider,
  - parse + validate the raw output into GeneratedAnswer (Pydantic),
  - retry once on invalid output, then raise a controlled GenerationError.

Context selection also lives here: reranked child chunks are expanded to
their parent text for generation when parent-child chunking is in use
(small precise chunks retrieve well; larger parents answer well).
"""
from __future__ import annotations

import asyncio
import json
import logging

from pydantic import BaseModel, Field, ValidationError

from app.providers.llm.base import LLMProvider
from app.schemas.models import Chunk

logger = logging.getLogger("voice_rag")


class GeneratedAnswer(BaseModel):
    answer: str = Field(..., min_length=1)
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class GenerationError(Exception):
    """Raised when no valid structured answer could be produced."""


PROMPT_TEMPLATE = """You are a grounded question-answering assistant.

Rules:
1. Answer ONLY using the supplied context.
2. If the context does not contain sufficient evidence, do not invent an answer.
3. In that case, state that the information is insufficient.
4. Return citations for factual claims as a list of chunk ids.
5. Respond with STRICT JSON only, matching exactly this schema:
   {{"answer": "<string>", "citations": ["<chunk_id>", ...], "confidence": <number between 0 and 1>}}
6. No markdown fences, no commentary outside the JSON.

Context passages:
{context}

Question: {query}

JSON response:"""


def build_prompt(query: str, context_chunks: list[Chunk]) -> str:
    lines = [
        f"[{chunk.chunk_id}] Title: {chunk.title}\n{chunk.text}"
        for chunk in context_chunks
    ]
    return PROMPT_TEMPLATE.format(
        context="\n\n".join(lines), query=query
    )


def select_context(
    reranked: list[tuple[Chunk, float]],
    chunks_by_id: dict[str, Chunk],
) -> list[Chunk]:
    """
    Expand reranked child chunks to their parent text for generation.
    Non-parent chunks pass through unchanged. Falls back to the child itself
    if its parent id cannot be resolved.
    """
    selected: list[Chunk] = []
    seen: set[str] = set()
    for chunk, _score in reranked:
        effective = chunk
        if chunk.parent_id:
            effective = chunks_by_id.get(chunk.parent_id, chunk)
        if effective.chunk_id not in seen:
            seen.add(effective.chunk_id)
            selected.append(effective)
    return selected


class RAGGenerator:
    def __init__(self, provider: LLMProvider, retries: int = 1) -> None:
        self.provider = provider
        self.retries = retries  # §17: retry once on invalid structured output.

    async def generate(self, query: str, context_chunks: list[Chunk]) -> GeneratedAnswer:
        prompt = build_prompt(query, context_chunks)
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                raw = await self.provider.generate(prompt)
                return self._validate(raw)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Invalid LLM output (attempt %d/%d): %s",
                    attempt + 1,
                    self.retries + 1,
                    exc,
                )
        raise GenerationError(f"invalid LLM response: {last_error}") from last_error

    async def generate_with_timeout(
        self, query: str, context_chunks: list[Chunk], timeout_s: float
    ) -> GeneratedAnswer:
        try:
            return await asyncio.wait_for(
                self.generate(query, context_chunks), timeout=timeout_s
            )
        except asyncio.TimeoutError as exc:
            raise GenerationError("LLM timed out") from exc

    @staticmethod
    def _validate(raw: str) -> GeneratedAnswer:
        """Parse strict JSON out of possibly noisy LLM output."""
        text = raw.strip()
        if text.startswith("```"):
            # Tolerate markdown fences even though the prompt forbids them.
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object found in LLM output")
        data = json.loads(text[start : end + 1])
        return GeneratedAnswer.model_validate(data)
