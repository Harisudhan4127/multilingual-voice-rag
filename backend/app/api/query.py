"""
Query endpoint (Phase 7): the text path into the RAG pipeline.

Request/response schema unchanged since Phase 1 -- only what happens behind
the endpoint changed (stub -> full orchestrated pipeline).
"""
from fastapi import APIRouter, Request

from app.pipeline.orchestrator import PipelineOrchestrator
from app.schemas.models import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    orchestrator: PipelineOrchestrator | None = getattr(
        request.app.state, "orchestrator", None
    )
    if orchestrator is None:
        return QueryResponse(
            request_id="unavailable",
            answer="Service is starting up; pipeline is not ready yet.",
            confidence=0.0,
            grounded=False,
            latency_ms=0.0,
            status="not_ready",
            error="pipeline not initialized",
        )
    return await orchestrator.run(payload.query)
