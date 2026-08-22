"""Factory mapping strategy name -> Chunker instance."""
from app.chunking.base import Chunker
from app.chunking.parent_child import ParentChildChunker
from app.chunking.semantic import SemanticChunker
from app.chunking.sentence import SentenceChunker
from app.chunking.token import TokenChunker

STRATEGY_REGISTRY: dict[str, type[Chunker]] = {
    "sentence": SentenceChunker,
    "token": TokenChunker,
    "semantic": SemanticChunker,
    "parent_child": ParentChildChunker,
}


def get_chunker(strategy: str, **kwargs) -> Chunker:
    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[strategy](**kwargs)


def all_strategies() -> list[str]:
    return list(STRATEGY_REGISTRY.keys())
