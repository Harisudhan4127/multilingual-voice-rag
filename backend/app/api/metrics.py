"""
Metrics endpoint (Phase 7): process-local counters and latency aggregates.
"""
from fastapi import APIRouter, Request

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def get_metrics(request: Request) -> dict:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None or getattr(orchestrator, "metrics", None) is None:
        return {"error": "pipeline not initialized"}
    return orchestrator.metrics.snapshot()
