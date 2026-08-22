"""
Guardrails (Phase 6 / Section 19).

Six checks wrap the pipeline:

  A  off-topic      - pre-retrieval: query shares nothing with corpus vocab
  B  unsafe         - pre-generation: request matches an unsafe-intent pattern
  C  retrieval      - post-rerank: best relevance below configured threshold
  D  grounding      - post-generation: answer sentences must be supported by
                     the retrieved context (token-level evidence)
  E  citations      - citations must reference actually-retrieved chunk ids
  F  hallucination  - consequence handler: any D/E failure swaps the answer
                     for the canonical refusal, never invented citations

All checks return GuardrailResult records so the orchestrator can time,
log and surface them without knowing each rule's internals. Heuristics are
deliberately simple and documented; they are honest baselines, not magic.
"""
from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from app.config import Settings
from app.indexing.bm25_store import tokenize
from app.pipeline.generator import GeneratedAnswer
from app.schemas.models import REFUSAL_ANSWER, Chunk

logger = logging.getLogger("voice_rag")


class GuardrailResult(BaseModel):
    guardrail: str
    passed: bool
    reason: str | None = None


# --- Guardrail B: unsafe-intent patterns ------------------------------------
# Demo-grade keyword heuristics, not a moderation system. Kept explicit here
# rather than hidden behind config so reviewers can audit every rule.
_UNSAFE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("weapons_explosives", re.compile(
        r"\b(make|build|assemble|construct|synthesize|detonate)\b.{0,40}\b"
        r"(bomb|explosive|ied|grenade|pipe bomb|nerve agent|sarin)\b", re.I)),
    ("weapons_firearms", re.compile(
        r"\b(gun|firearm|rifle|pistol)\b.{0,30}\b(illegal|untraceable|ghost)\b", re.I)),
    ("malware", re.compile(
        r"\b(write|create|deploy)\b.{0,30}\b(malware|ransomware|keylogger|botnet|virus)\b", re.I)),
    ("hacking", re.compile(
        r"\b(hack|hijack|breach)\b.{0,30}\b(account|bank|email|server|database)\b", re.I)),
    ("self_harm", re.compile(
        r"\b(how (to|do i)|ways to)\b.{0,20}\b(kill myself|commit suicide|self[- ]harm)\b", re.I)),
    ("violence", re.compile(
        r"\b(how (to|do i))\b.{0,20}\b(kill|harm|poison|kidnap)\b\s+(someone|a person|people|my)", re.I)),
    ("csae", re.compile(r"\b(child|csam|cp)\b.{0,25}\b(pornograph|sexual|exploit)\b", re.I)),
    ("drugs_synthesis", re.compile(
        r"\b(synthesize|manufacture|cook)\b.{0,25}\b(meth|methamphetamine|mdma|fentanyl)\b", re.I)),
)

# Words too generic to count as topical evidence for Guardrail A.
_STOPWORDS: frozenset[str] = frozenset({
    "what", "who", "when", "where", "why", "how", "which", "is", "are", "was",
    "were", "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "do",
    "does", "did", "can", "could", "should", "would", "will", "i", "you", "it",
    "this", "that", "these", "those", "with", "about", "explain", "tell",
    "me", "please", "give", "some", "information",
})


def _content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 2}


class Guardrails:
    def __init__(self, settings: Settings, corpus_vocab: set[str]) -> None:
        self.settings = settings
        self.corpus_vocab = corpus_vocab

    # -- A: off-topic --------------------------------------------------------

    def check_off_topic(self, query: str) -> GuardrailResult:
        query_tokens = _content_tokens(query)
        if not query_tokens:
            # Nothing topical left after stopword removal: e.g. pure filler.
            return GuardrailResult(
                guardrail="off_topic", passed=False,
                reason="query has no substantive terms",
            )
        hits = query_tokens & self.corpus_vocab
        min_hits = self.settings.OFF_TOPIC_MIN_VOCAB_HITS
        if len(hits) < min_hits:
            return GuardrailResult(
                guardrail="off_topic", passed=False,
                reason=(
                    f"query vocabulary does not intersect the dataset "
                    f"({sorted(query_tokens)} vs corpus)"
                ),
            )
        return GuardrailResult(guardrail="off_topic", passed=True)

    # -- B: unsafe -----------------------------------------------------------

    def check_unsafe(self, query: str) -> GuardrailResult:
        for label, pattern in _UNSAFE_PATTERNS:
            if pattern.search(query):
                return GuardrailResult(
                    guardrail="unsafe", passed=False,
                    reason=f"matched unsafe pattern '{label}'",
                )
        return GuardrailResult(guardrail="unsafe", passed=True)

    # -- C: retrieval confidence ----------------------------------------------

    def check_retrieval_confidence(
        self, reranked: list[tuple[Chunk, float]]
    ) -> GuardrailResult:
        if not reranked:
            return GuardrailResult(
                guardrail="retrieval_confidence", passed=False,
                reason="no documents retrieved",
            )
        best_score = max(score for _, score in reranked)
        threshold = self.settings.RETRIEVAL_CONFIDENCE_THRESHOLD
        if best_score < threshold:
            return GuardrailResult(
                guardrail="retrieval_confidence", passed=False,
                reason=f"best score {best_score:.3f} below threshold {threshold}",
            )
        return GuardrailResult(guardrail="retrieval_confidence", passed=True)

    # -- E: citation validation -----------------------------------------------

    def validate_citations(
        self, answer: GeneratedAnswer, retrieved_ids: set[str]
    ) -> tuple[GeneratedAnswer, GuardrailResult]:
        valid = [c for c in answer.citations if c in retrieved_ids]
        dropped = [c for c in answer.citations if c not in retrieved_ids]
        if dropped:
            logger.warning("Dropped non-retrieved citations: %s", dropped)
        cleaned = answer.model_copy(update={"citations": valid})

        if answer.citations and not valid:
            return cleaned, GuardrailResult(
                guardrail="citations", passed=False,
                reason=f"all citations invalid: {dropped}",
            )
        return cleaned, GuardrailResult(
            guardrail="citations",
            passed=True,
            reason=f"dropped {len(dropped)} invalid citation(s)" if dropped else None,
        )

    # -- D: grounding -----------------------------------------------------------

    @staticmethod
    def _sentence_supported(sentence: str, context_tokens: set[str],
                            threshold: float) -> bool:
        sent_tokens = _content_tokens(sentence)
        if not sent_tokens:
            return True  # punctuation-only fragments cannot be 'unsupported claims'
        overlap = len(sent_tokens & context_tokens) / len(sent_tokens)
        return overlap >= threshold

    def check_grounding(
        self, answer: GeneratedAnswer, context_chunks: list[Chunk]
    ) -> GuardrailResult:
        if answer.answer.strip() == REFUSAL_ANSWER:
            return GuardrailResult(guardrail="grounding", passed=True)

        context_tokens: set[str] = set()
        for chunk in context_chunks:
            context_tokens |= _content_tokens(chunk.text)

        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.answer) if s.strip()
        ]
        unsupported = [
            s for s in sentences
            if not self._sentence_supported(
                s, context_tokens, self.settings.GROUNDING_THRESHOLD
            )
        ]
        if unsupported:
            return GuardrailResult(
                guardrail="grounding", passed=False,
                reason=f"{len(unsupported)}/{len(sentences)} sentences unsupported "
                       f"(threshold={self.settings.GROUNDING_THRESHOLD})",
            )
        return GuardrailResult(guardrail="grounding", passed=True)


def apply_hallucination_guard(answer: GeneratedAnswer) -> GeneratedAnswer:
    """Guardrail F: hard swap to the canonical refusal. No invented citations."""
    return answer.model_copy(
        update={
            "answer": REFUSAL_ANSWER,
            "citations": [],
            "confidence": 0.0,
        }
    )
