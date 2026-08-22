"""
FastAPI application entrypoint.

Lifespan: every heavy component -- embedding model, Qdrant connection, BM25
index, reranker, LLM/STT providers, guardrails, orchestrator -- is built
exactly once at startup via app.dependencies.build_pipeline and attached to
app.state. Nothing in the request path loads a model or opens a connection.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, metrics, query, voice
from app.config import get_settings
from app.utils.logging import configure_logging

logger = logging.getLogger("voice_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.dependencies import build_pipeline

    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    logger.info("Starting voice-rag backend | env=%s", settings.APP_ENV)

    components = await build_pipeline(settings)

    app.state.settings = settings
    app.state.embedder = components.embedder
    app.state.vector_store = components.vector_store
    app.state.bm25_store = components.bm25_store
    app.state.reranker = components.reranker
    app.state.stt_provider = components.stt_provider
    app.state.orchestrator = components.orchestrator

    logger.info(
        "Startup complete | embedder=%s(%s) vector_mode=%s bm25_chunks=%d "
        "reranker=%s llm=%s stt=%s",
        components.embedder.kind,
        components.embedder.model_name,
        components.vector_store.mode,
        len(components.bm25_store),
        components.reranker.kind,
        components.llm_provider.name,
        components.stt_provider.name,
    )

    yield

    logger.info("Shutting down voice-rag backend")
    await components.close()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Voice-Enabled RAG API",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(voice.router, prefix="/api/v1")
    app.include_router(metrics.router, prefix="/api/v1")

    return app


app = create_app()
