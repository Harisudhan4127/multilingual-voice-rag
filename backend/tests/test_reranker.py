"""Tests for configurable reranker selection via RERANKER_TYPE."""
import pytest

from app.config import Settings
from app.pipeline.reranker import CrossEncoderReranker, LexicalReranker, get_reranker


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


class TestRerankerTypeConfig:
    def test_default_is_cross_encoder(self):
        s = _settings()
        assert s.RERANKER_TYPE == "cross_encoder"

    def test_cross_encoder_explicit(self):
        s = _settings(RERANKER_TYPE="cross_encoder")
        assert s.RERANKER_TYPE == "cross_encoder"

    def test_lexical_explicit(self):
        s = _settings(RERANKER_TYPE="lexical")
        assert s.RERANKER_TYPE == "lexical"

    def test_invalid_type_rejected(self):
        with pytest.raises(Exception):
            _settings(RERANKER_TYPE="invalid")


class TestGetReranker:
    def test_lexical_mode_returns_lexical_reranker(self):
        s = _settings(RERANKER_TYPE="lexical")
        reranker = get_reranker(s)
        assert isinstance(reranker, LexicalReranker)
        assert reranker.kind == "lexical"

    def test_cross_encoder_mode_returns_cross_encoder(self):
        s = _settings(RERANKER_TYPE="cross_encoder")
        reranker = get_reranker(s)
        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker.kind == "cross_encoder"

    def test_default_mode_returns_cross_encoder(self):
        s = _settings()
        reranker = get_reranker(s)
        assert isinstance(reranker, CrossEncoderReranker)

    def test_cross_encoder_fallback_to_lexical(self, monkeypatch):
        """If CrossEncoder import fails, fallback to LexicalReranker."""
        import app.pipeline.reranker as mod

        def _boom(model_name):
            raise RuntimeError("no torch")

        monkeypatch.setattr(mod, "CrossEncoderReranker", lambda m: _boom(m))
        s = _settings(RERANKER_TYPE="cross_encoder")
        reranker = get_reranker(s)
        assert isinstance(reranker, LexicalReranker)
