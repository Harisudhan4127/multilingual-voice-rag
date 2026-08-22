"""
MockLLMProvider (Phase 5): deterministic, offline, extractive.

Given a RAG prompt containing numbered [chunk_id]-tagged context passages,
it "answers" exactly like a well-behaved grounded model would:
  - scores context sentences by lexical overlap with the query,
  - copies the best-scoring sentences verbatim (never invents content),
  - cites the chunk ids those sentences came from,
  - outputs STRICT JSON so the downstream parser/validation path is the same
    one a real LLM goes through.

If nothing in the context overlaps the query, it returns the refusal shape
(empty citations, low confidence) -- exercising guardrails end-to-end.
"""
from __future__ import annotations

import json
import re

from app.indexing.bm25_store import tokenize
from app.providers.llm.base import LLMProvider
from app.schemas.models import REFUSAL_ANSWER

_CHUNK_TAG_RE = re.compile(r"\[([^\]]+)\]\s*(.+)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


class MockLLMProvider(LLMProvider):
    name = "mock"

    async def generate(self, prompt: str) -> str:
        query, context = self._parse_prompt(prompt)
        query_tokens = set(tokenize(query))

        scored_sentences: list[tuple[float, int, str, str]] = []
        for order, (chunk_id, text) in enumerate(context):
            for sentence in split_sentences(text):
                sent_tokens = tokenize(sentence)
                if not sent_tokens:
                    continue
                overlap = len(query_tokens & set(sent_tokens))
                score = overlap / (len(sent_tokens) ** 0.5)
                if overlap > 0:
                    scored_sentences.append((score, order, sentence, chunk_id))

        if not scored_sentences:
            return json.dumps(
                {
                    "answer": REFUSAL_ANSWER,
                    "citations": [],
                    "confidence": 0.05,
                }
            )

        # Best sentences across chunks; keep source order for readability.
        top = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:2]
        top.sort(key=lambda x: (x[1], x[2]))
        answer_parts: list[str] = []
        cited: list[str] = []
        for _, _, sentence, chunk_id in top:
            if sentence not in answer_parts:
                answer_parts.append(sentence)
                if chunk_id not in cited:
                    cited.append(chunk_id)

        mean_score = sum(s for s, *_ in top) / len(top)
        # Overlap ratio of query terms actually covered by the answer.
        covered = set()
        for sentence in answer_parts:
            covered |= set(tokenize(sentence))
        coverage = len(query_tokens & covered) / max(len(query_tokens), 1)
        confidence = min(0.95, round(0.4 * mean_score + 0.6 * coverage, 2))

        return json.dumps(
            {
                "answer": " ".join(answer_parts),
                "citations": cited,
                "confidence": confidence,
            }
        )

    @staticmethod
    def _parse_prompt(prompt: str) -> tuple[str, list[tuple[str, str]]]:
        """Extract the question line and [(chunk_id, passage)] pairs."""
        query = ""
        context: list[tuple[str, str]] = []
        current_id: str | None = None
        current_text: list[str] = []

        def flush() -> None:
            nonlocal current_id, current_text
            if current_id is not None:
                context.append((current_id, " ".join(current_text).strip()))
            current_id, current_text = None, []

        for line in prompt.splitlines():
            tag = _CHUNK_TAG_RE.match(line.strip())
            if tag:
                flush()
                current_id, current_text = tag.group(1), [tag.group(2)]
            elif line.startswith("Question:"):
                query = line.split(":", 1)[1].strip()
            elif current_id is not None and line.strip():
                current_text.append(line.strip())
        flush()

        if not query:
            # Fallback: first non-context line mentioning a question mark.
            match = re.search(r"^(.*\?)\s*$", prompt, flags=re.MULTILINE)
            query = match.group(1) if match else ""
        return query, context
