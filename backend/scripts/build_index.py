"""
Build the full retrieval index:

  documents -> chunks -> embeddings -> Qdrant (+ persisted chunks JSONL)

BM25 needs no build step: the in-memory index is rebuilt from the same
chunks JSONL in milliseconds at application startup.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --strategy parent_child
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking.factory import all_strategies  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.indexing.build import build_index  # noqa: E402
from app.indexing.embeddings import get_embedder  # noqa: E402
from app.indexing.qdrant_store import QdrantVectorStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=all_strategies(),
        default=None,
        help="Chunking strategy (default: CHUNKING_STRATEGY from .env)",
    )
    args = parser.parse_args()

    settings = get_settings()
    embedder = get_embedder(settings)
    vector_store = QdrantVectorStore(settings)

    async def run() -> None:
        report = await build_index(
            settings,
            embedder,
            vector_store,
            strategy=args.strategy,
        )
        print("\n--- Index build report ---")
        for key, value in report.__dict__.items():
            print(f"{key:20s} {value}")

    try:
        asyncio.run(run())
    finally:
        asyncio.run(vector_store.close())


if __name__ == "__main__":
    main()
