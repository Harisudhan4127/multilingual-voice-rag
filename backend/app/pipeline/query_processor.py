"""
Query processing (Section 14).

Scope is deliberately conservative:
  - whitespace normalization
  - speech-artifact cleanup (filler sounds that leak through STT)
  - script-based language detection (heuristic, practical subset)
  - basic classification (question vs keyword lookup)
  - empty-query validation

It does NOT rewrite, expand or "improve" the user's question -- aggressive
query rewriting changes retrieval semantics and makes evaluation results
untrustworthy.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

# Filler sounds commonly produced by ASR around silence / hesitation.
_SPEECH_ARTIFACT_RE = re.compile(
    r"\b(u+h+|u+m+|e+r+|a+h+|h+m+|o+h+|yeah|okay|ok|so+|well)\b[,.]?\s*",
    flags=re.IGNORECASE,
)
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_REPEATED_PUNCT_RE = re.compile(r"([!?.]){2,}")

# Script detection: a small set of Unicode ranges covering the languages
# MSMARCO-XI targets. Latin/default falls back to "en".
_SCRIPT_RANGES: tuple[tuple[str, str], ...] = (
    ("hi", "\u0900-\u097F"),  # Devanagari (Hindi)
    ("bn", "\u0980-\u09FF"),  # Bengali
    ("ta", "\u0B80-\u0BFF"),  # Tamil
    ("te", "\u0C00-\u0C7F"),  # Telugu
    ("mr", "\u0900-\u097F"),  # Marathi shares Devanagari; kept for clarity
)

_QUESTION_STARTERS = (
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "which",
    "whose",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "explain",
    "describe",
    "define",
    "list",
    "compare",
)


class ProcessedQuery(BaseModel):
    original: str
    cleaned: str
    language: str = "en"
    query_type: str = "question"  # question | keyword
    valid: bool = True
    invalid_reason: str | None = None


def detect_language(text: str) -> str:
    """Return the first matching Indic script code, else 'en'."""
    for lang, pattern in _SCRIPT_RANGES:
        if re.search(f"[{pattern}]", text):
            return lang
    return "en"


class QueryProcessor:
    def process(self, raw_query: str) -> ProcessedQuery:
        original = raw_query.strip()

        text = _SPEECH_ARTIFACT_RE.sub(" ", original)
        text = _MULTI_SPACE_RE.sub(" ", text).strip()
        text = _REPEATED_PUNCT_RE.sub(r"\1", text)

        if not text or not any(ch.isalnum() for ch in text):
            return ProcessedQuery(
                original=original,
                cleaned=text,
                valid=False,
                invalid_reason="empty query",
            )

        lowered = text.lower()
        starts_like_question = lowered.split()[0] in _QUESTION_STARTERS
        has_question_mark = text.endswith("?")
        query_type = (
            "question"
            if (starts_like_question or has_question_mark)
            else "keyword"
        )

        return ProcessedQuery(
            original=original,
            cleaned=text,
            language=detect_language(text),
            query_type=query_type,
            valid=True,
        )
