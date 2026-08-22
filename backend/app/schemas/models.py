"""
Shared Pydantic models for API requests/responses and internal pipeline state.

This file grows across phases. Phase 1 only needs health + a stub query
contract so the frontend has something real to talk to.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    app_env: str
    version: str = "0.1.0"
    components: dict[str, str] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


# --- Dataset layer (Phase 2) ---


class Document(BaseModel):
    """A single source document/passage, independent of chunking strategy."""

    document_id: str
    passage_id: str
    title: str
    text: str
    language: str = "en"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A chunk produced by some Chunker implementation from a Document."""

    document_id: str
    chunk_id: str
    strategy: str
    position: int
    total_chunks: int
    language: str
    title: str
    text: str
    # Parent-child strategy only: the child's parent_id references the
    # larger parent chunk that should be used as LLM context once this
    # (smaller, more precise) child chunk is retrieved. None for all other
    # strategies, and None on parent chunks themselves.
    parent_id: str | None = None
    is_parent: bool = False


class SourceCitation(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    text: str
    score: float


# The single canonical refusal answer used by guardrails, the mock LLM and
# the orchestrator (Section 19 / Guardrail F). Never invented anywhere else.
REFUSAL_ANSWER: str = (
    "I don't have enough information in the provided dataset to answer this reliably."
)


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    # chunk ids the generator actually cited (validated against retrieval);
    # distinct from `sources`, which lists everything reranked into context.
    citations: list[str] = Field(default_factory=list)
    sources: list[SourceCitation] = Field(default_factory=list)
    confidence: float
    grounded: bool
    latency_ms: float
    timings: dict[str, float] = Field(default_factory=dict)
    status: str = "ok"
    error: str | None = None
