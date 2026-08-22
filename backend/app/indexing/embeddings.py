"""
Embedding models (Phase 3).

Loads a local sentence-transformers model exactly once (at application
startup or index build time) and exposes batch embedding. Never loaded
per-request.

Fallback chain, each step loudly logged:
  1. settings.EMBEDDING_MODEL        (primary, e.g. multilingual MiniLM)
  2. settings.EMBEDDING_FALLBACK_MODEL (smaller English MiniLM)
  3. HashingEmbedder                 (deterministic feature hashing; keeps the
     system runnable on a machine with no model downloads at all -- degraded
     retrieval quality, but honest about it and clearly flagged in logs and
     the health endpoint)
"""
from __future__ import annotations

import hashlib
import logging
import re

import numpy as np

from app.config import Settings

logger = logging.getLogger("voice_rag")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class BaseEmbedder:
    """Common embedder interface. All vectors are L2-normalized float32."""

    model_name: str = "base"
    dim: int = 0
    kind: str = "base"  # "sentence_transformers" | "hashing"

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity for L2-normalized vectors is just the dot product."""
        return float(np.clip(np.dot(a, b), -1.0, 1.0))


class SentenceTransformerEmbedder(BaseEmbedder):
    kind = "sentence_transformers"

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        # Imported here so that importing app.indexing.embeddings does not pay
        # the multi-second torch/sentence-transformers import cost unless an
        # instance is actually constructed.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        # sentence-transformers 6.x renamed get_sentence_embedding_dimension.
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder(BaseEmbedder):
    """
    Dependency-free deterministic embedder: feature hashing over unigrams +
    bigrams with a signed hash trick, then L2 normalization.

    Quality is far below a real transformer model (no semantics, only lexical
    overlap) but it is stable, instant, needs no downloads, and lets the full
    pipeline be exercised end-to-end anywhere. The health endpoint reports
    kind="hashing" so this mode can never pass unnoticed.
    """

    kind = "hashing"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.model_name = f"hashing-{dim}"

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return tokens + bigrams

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feat in self._features(text):
                digest = hashlib.md5(feat.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[i, idx] += sign
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out


def get_embedder(settings: Settings) -> BaseEmbedder:
    """Build the configured embedder with graceful fallbacks. Logs every step."""
    for name in (settings.EMBEDDING_MODEL, settings.EMBEDDING_FALLBACK_MODEL):
        if not name:
            continue
        try:
            embedder = SentenceTransformerEmbedder(name)
            logger.info("Loaded embedding model '%s' (dim=%d)", name, embedder.dim)
            return embedder
        except Exception as exc:  # noqa: BLE001 - fallback is the point here
            logger.warning("Could not load embedding model '%s': %s", name, exc)

    if not settings.ALLOW_HASHING_EMBEDDER_FALLBACK:
        raise RuntimeError(
            "No embedding model could be loaded and "
            "ALLOW_HASHING_EMBEDDER_FALLBACK=false"
        )

    logger.warning(
        "Falling back to HashingEmbedder(dim=%d). Retrieval quality is "
        "DEGRADED; install sentence-transformers + torch or pre-download "
        "'%s' to fix.",
        settings.EMBEDDING_DIM,
        settings.EMBEDDING_MODEL,
    )
    return HashingEmbedder(settings.EMBEDDING_DIM)
