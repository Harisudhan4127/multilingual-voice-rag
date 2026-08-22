"""
Health endpoint. In Phase 1 this only reports app liveness + config.
Later phases add real component checks (Qdrant, embedding model, etc.)
via app.state, without changing this router's shape.
"""
from fastapi import APIRouter, Request

from app.config import get_settings
from app.schemas.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = get_settings()

    components: dict[str, str] = {"api": "ok"}

    # Later phases populate these via app.state (set at startup).
    # We check hasattr defensively so Phase 1 doesn't need them yet.
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store is not None:
        components["vector_store"] = await vector_store.health_check()
    else:
        components["vector_store"] = "not_initialized"

    embedder = getattr(request.app.state, "embedder", None)
    components["embedder"] = "ok" if embedder is not None else "not_initialized"

    overall = "ok" if all(v in ("ok",) or v == "not_initialized" for v in components.values()) else "degraded"

    return HealthResponse(
        status=overall,
        app_env=settings.APP_ENV,
        components=components,
    )
