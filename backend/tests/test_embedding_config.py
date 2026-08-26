"""Tests for configurable embedding model selection."""
import pytest

from app.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


class TestEmbeddingConfig:
    def test_default_model_is_multilingual(self):
        s = _settings()
        assert s.EMBEDDING_MODEL == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def test_default_fallback_is_lightweight(self):
        s = _settings()
        assert s.EMBEDDING_FALLBACK_MODEL == "sentence-transformers/all-MiniLM-L6-v2"

    def test_embedding_dim_default(self):
        s = _settings()
        assert s.EMBEDDING_DIM == 384

    def test_lightweight_model_selectable_via_env(self):
        s = _settings(EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2")
        assert s.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"

    def test_fallback_model_configurable(self):
        s = _settings(EMBEDDING_FALLBACK_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        assert s.EMBEDDING_FALLBACK_MODEL == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def test_hashing_fallback_toggleable(self):
        s = _settings(ALLOW_HASHING_EMBEDDER_FALLBACK=False)
        assert s.ALLOW_HASHING_EMBEDDER_FALLBACK is False

    def test_auto_index_default_is_true(self):
        s = _settings()
        assert s.AUTO_INDEX_ON_STARTUP is True

    def test_auto_index_can_be_disabled(self):
        s = _settings(AUTO_INDEX_ON_STARTUP=False)
        assert s.AUTO_INDEX_ON_STARTUP is False


class TestEmbeddingModelDimensions:
    """Both supported models produce 384-dim vectors. Verify config stays aligned."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
        ],
    )
    def test_model_dim_matches_config(self, model_name):
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(model_name)
        get_dim = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
        dim = int(get_dim())
        assert dim == 384, f"{model_name} produces {dim}-dim, expected 384"


class TestQdrantDimensionHandling:
    """ensure_collection recreates the collection when the dimension changes."""

    def test_collection_recreated_on_dim_change(self, tmp_path):
        """Simulate switching from 384-dim model to 32-dim model."""
        from app.config import Settings
        from app.indexing.qdrant_store import QdrantVectorStore

        settings = Settings(
            QDRANT_URL="http://localhost:1",  # nothing listens here
            QDRANT_LOCAL_PATH=str(tmp_path / "qdrant"),
            QDRANT_COLLECTION="test_dim_change",
            _env_file=None,
        )
        store = QdrantVectorStore(settings)
        client = store._require_client()

        import asyncio
        asyncio.run(store.ensure_collection(384))
        info_384 = client.get_collection("test_dim_change")
        assert info_384.config.params.vectors.size == 384

        asyncio.run(store.ensure_collection(32))  # dim changed -> recreated
        info_32 = client.get_collection("test_dim_change")
        assert info_32.config.params.vectors.size == 32

        asyncio.run(store.close())


class TestAutoIndexPrevention:
    """AUTO_INDEX_ON_STARTUP=false must prevent index building."""

    def test_ensure_index_skips_when_disabled(self, tmp_path):
        import asyncio

        from app.config import Settings
        from app.indexing.embeddings import HashingEmbedder
        from app.indexing.qdrant_store import QdrantVectorStore

        settings = Settings(
            AUTO_INDEX_ON_STARTUP=False,
            QDRANT_URL="http://localhost:1",
            QDRANT_LOCAL_PATH=str(tmp_path / "qdrant_auto"),
            QDRANT_COLLECTION="test_no_auto",
            PROCESSED_CHUNKS_FILE=str(tmp_path / "nonexistent.jsonl"),
            _env_file=None,
        )

        class FakeEmbedder:
            kind = "fake"
            model_name = "fake"
            dim = 64

            def embed(self, texts):
                import numpy as np
                return np.zeros((len(texts), self.dim), dtype=np.float32)

        store = QdrantVectorStore(settings)
        embedder = FakeEmbedder()

        from app.dependencies import ensure_index

        chunks = asyncio.run(ensure_index(settings, embedder, store))
        # No chunks should exist (no file, no auto-index)
        assert chunks == []
        asyncio.run(store.close())
