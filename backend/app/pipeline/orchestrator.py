"""
Pipeline orchestrator (Phase 7 / Sections 20-21).

The single structured path every request takes:

    Request -> validate -> query processing -> input guardrails (A/B)
            -> cache? -> dense + BM25 (parallel) -> RRF fusion
            -> reranker -> retrieval-confidence guardrail (C)
            -> context selection -> LLM (retry + timeout)
            -> citation guardrail (E) -> grounding guardrail (D/F)
            -> response

Stage contract: every stage is individually timed into state.timings, every
exception becomes a controlled PipelineState.status -- a client never sees a
raw stack trace (unless APP_ENV=development, which appends stage errors for
debuggability). Retries: the generator retries invalid LLM output once
internally; this orchestrator adds one provider-level retry on transient
failures. No stage may hang past its configured timeout.
"""
from __future__ import annotations

import asyncio
import logging
import time

from pydantic import BaseModel, Field

from app.config import Settings
from app.pipeline.generator import GeneratedAnswer, GenerationError, RAGGenerator
from app.pipeline.guardrails import Guardrails, apply_hallucination_guard
from app.pipeline.hybrid_search import HybridRetriever
from app.pipeline.query_processor import ProcessedQuery, QueryProcessor
from app.pipeline.reranker import BaseReranker
from app.schemas.models import (
    REFUSAL_ANSWER,
    Chunk,
    QueryResponse,
    SourceCitation,
)
from app.utils.cache import TTLCache
from app.utils.timing import timed

logger = logging.getLogger("voice_rag")

# Statuses that count as refusals for metrics/reporting.
_REFUSAL_STATUSES = {
    "refused_unsafe",
    "refused_off_topic",
    "refused_low_confidence",
    "grounding_failed",
}


class PipelineState(BaseModel):
    request_id: str
    query: str
    transcript: str | None = None  # set on the voice path
    processed: ProcessedQuery | None = None
    retrieval_results: list[tuple[Chunk, float]] = Field(default_factory=list)
    reranked_results: list[tuple[Chunk, float]] = Field(default_factory=list)
    context: list[Chunk] = Field(default_factory=list)
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    grounded: bool = False
    timings: dict[str, float] = Field(default_factory=dict)
    status: str = "started"
    error: str | None = None
    cache_hit: bool = False


class PipelineOrchestrator:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        reranker: BaseReranker,
        generator: RAGGenerator,
        guardrails: Guardrails,
        chunks_by_id: dict[str, Chunk],
        metrics=None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.guardrails = guardrails
        self.chunks_by_id = chunks_by_id
        self.metrics = metrics
        self.processor = QueryProcessor()
        self.response_cache = TTLCache(maxsize=256, ttl_seconds=120.0)

    # ------------------------------------------------------------------ #

    async def run(self, raw_query: str, transcript: str | None = None) -> QueryResponse:
        start = time.perf_counter()
        import uuid

        state = PipelineState(
            request_id=str(uuid.uuid4()), query=raw_query, transcript=transcript
        )
        if self.metrics:
            self.metrics.record_request(voice=transcript is not None)

        try:
            await self._pipeline(state)
        except Exception as exc:  # noqa: BLE001 - last-resort controlled failure
            logger.exception("Unhandled pipeline error")
            state.status = "internal_error"
            state.error = str(exc)
            state.answer = REFUSAL_ANSWER

        total_ms = round((time.perf_counter() - start) * 1000, 2)
        state.timings["total_ms"] = total_ms

        return self._build_response(state)

    # ------------------------------------------------------------------ #

    async def _pipeline(self, state: PipelineState) -> None:
        s = self.settings

        with timed(state.timings, "query_processing_ms"):
            processed = self.processor.process(state.query)
            state.processed = processed
        if not processed.valid:
            state.status = "invalid_query"
            state.answer = REFUSAL_ANSWER
            state.error = processed.invalid_reason
            return

        with timed(state.timings, "input_guardrails_ms"):
            off_topic = self.guardrails.check_off_topic(processed.cleaned)
            unsafe = self.guardrails.check_unsafe(processed.cleaned)
        if not unsafe.passed:
            state.status = "refused_unsafe"
            state.answer = REFUSAL_ANSWER
            state.error = f"unsafe: {unsafe.reason}"
            logger.warning("Refused unsafe query (%s)", state.request_id)
            return
        if not off_topic.passed:
            state.status = "refused_off_topic"
            state.answer = REFUSAL_ANSWER
            state.error = f"off-topic: {off_topic.reason}"
            logger.info("Refused off-topic query (%s)", state.request_id)
            return

        cache_key = f"{processed.cleaned.lower()}|{s.CHUNKING_STRATEGY}"
        cached = self.response_cache.get(cache_key)
        if cached is not None:
            state.cache_hit = True
            state.retrieval_results = cached["retrieval_results"]
            state.reranked_results = cached["reranked_results"]
            state.context = cached["context"]
            state.answer = cached["answer"].answer
            state.citations = cached["answer"].citations
            state.confidence = cached["answer"].confidence
            state.grounded = True
            state.status = "ok"
            state.timings["retrieval_ms"] = 0.0
            state.timings["rerank_ms"] = 0.0
            state.timings["generation_ms"] = 0.0
            return

        # --- Retrieval (dense + BM25 fused, parallel inside) ---------------
        with timed(state.timings, "retrieval_ms"):
            try:
                state.retrieval_results = await asyncio.wait_for(
                    self.retriever.search(
                        processed.cleaned, top_k=max(s.TOP_K_DENSE, s.TOP_K_BM25)
                    ),
                    timeout=s.RETRIEVAL_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                state.status = "retrieval_error"
                state.error = "hybrid retrieval timed out"
                state.answer = REFUSAL_ANSWER
                logger.error("Retrieval timeout (%s)", state.request_id)
                return
            except RuntimeError as exc:
                state.status = "retrieval_error"
                state.error = f"retrieval unavailable: {exc}"
                state.answer = REFUSAL_ANSWER
                logger.error("Retrieval failure (%s): %s", state.request_id, exc)
                return
        if not state.retrieval_results:
            state.status = "refused_no_results"
            state.answer = REFUSAL_ANSWER
            return

        # --- Reranking -------------------------------------------------------
        with timed(state.timings, "rerank_ms"):
            try:
                state.reranked_results = await asyncio.wait_for(
                    self.reranker.rerank(
                        processed.cleaned,
                        state.retrieval_results,
                        top_k=s.TOP_K_RERANK,
                    ),
                    timeout=s.RETRIEVAL_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                # Degrade gracefully: keep fused order instead of failing.
                logger.warning("Rerank timeout; keeping fusion order")
                state.reranked_results = state.retrieval_results[: s.TOP_K_RERANK]
                state.timings["rerank_degraded"] = 1.0

        confidence_check = self.guardrails.check_retrieval_confidence(
            state.reranked_results
        )
        if not confidence_check.passed:
            state.status = "refused_low_confidence"
            state.answer = REFUSAL_ANSWER
            state.error = confidence_check.reason
            return

        # --- Context selection ----------------------------------------------
        from app.pipeline.generator import select_context

        with timed(state.timings, "context_selection_ms"):
            state.context = select_context(state.reranked_results, self.chunks_by_id)

        # --- Generation (provider-level retry once) --------------------------
        with timed(state.timings, "generation_ms"):
            answer = await self._generate_with_recovery(state)
        if answer is None:
            return  # status already set by _generate_with_recovery

        state.answer = answer.answer
        state.citations = answer.citations
        state.confidence = answer.confidence

        # --- Post-generation guardrails --------------------------------------
        retrieved_ids = {chunk.chunk_id for chunk, _ in state.retrieval_results}
        with timed(state.timings, "output_guardrails_ms"):
            cleaned_answer, citation_result = self.guardrails.validate_citations(
                answer, retrieved_ids
            )
            grounding_result = self.guardrails.check_grounding(
                cleaned_answer, state.context
            )

        if not citation_result.passed or not grounding_result.passed:
            safe = apply_hallucination_guard(cleaned_answer)
            state.answer = safe.answer
            state.citations = []
            state.confidence = 0.0
            state.grounded = False
            state.status = "grounding_failed"
            state.error = grounding_result.reason or citation_result.reason
            return

        state.answer = cleaned_answer.answer
        state.citations = cleaned_answer.citations
        state.grounded = True
        state.status = "ok"

        self.response_cache.set(
            cache_key,
            {
                "retrieval_results": state.retrieval_results,
                "reranked_results": state.reranked_results,
                "context": state.context,
                "answer": cleaned_answer,
            },
        )

    async def _generate_with_recovery(self, state: PipelineState) -> GeneratedAnswer | None:
        """One controlled provider-level retry after generator-internal retry."""
        attempts = 2
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return await self.generator.generate_with_timeout(
                    state.processed.cleaned,
                    state.context,
                    timeout_s=self.settings.LLM_TIMEOUT_S,
                )
            except GenerationError as exc:
                # Invalid output was already retried inside the generator;
                # a second identical attempt would only burn latency.
                state.status = "generation_error"
                state.error = str(exc)
                state.answer = REFUSAL_ANSWER
                logger.error("Generation failed (%s): %s", state.request_id, exc)
                return None
            except Exception as exc:  # noqa: BLE001 - transient provider issues
                last_exc = exc
                logger.warning(
                    "LLM provider attempt %d/%d failed (%s): %s",
                    attempt + 1,
                    attempts,
                    state.request_id,
                    exc,
                )
                await asyncio.sleep(0.1 * (attempt + 1))
        state.status = "generation_error"
        state.error = f"LLM provider failed after {attempts} attempts: {last_exc}"
        state.answer = REFUSAL_ANSWER
        return None

    # ------------------------------------------------------------------ #

    def _build_response(self, state: PipelineState) -> QueryResponse:
        sources = [
            SourceCitation(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.title,
                text=chunk.text[:400],
                score=round(score, 4),
            )
            for chunk, score in state.reranked_results[: self.settings.TOP_K_RERANK]
        ]
        # Sources are only shown for successful grounded answers.
        if state.status != "ok":
            sources = []

        if self.metrics:
            self.metrics.record_response(
                status=state.status,
                latency_ms=state.timings.get("total_ms", 0.0),
                timings=state.timings,
                refused=state.status in _REFUSAL_STATUSES,
                errored=state.status.endswith("_error") or state.status == "invalid_query",
                cache_hit=state.cache_hit,
            )

        error = state.error
        if error and self.settings.APP_ENV != "development":
            # Never leak internals outside development.
            error = state.status

        logger.info(
            "request=%s status=%s total=%.1fms stages=%s",
            state.request_id,
            state.status,
            state.timings.get("total_ms", 0.0),
            {k: v for k, v in state.timings.items() if k != "total_ms"},
        )

        return QueryResponse(
            request_id=state.request_id,
            answer=state.answer or "",
            citations=state.citations,
            sources=sources,
            confidence=round(state.confidence, 3),
            grounded=state.grounded,
            latency_ms=state.timings.get("total_ms", 0.0),
            timings=state.timings,
            status=state.status,
            error=error,
        )
