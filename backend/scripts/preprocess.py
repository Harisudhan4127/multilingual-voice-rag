"""
Preprocess: load the configured dataset, chunk it with the configured (or
given) strategy, and persist chunks to data/processed/chunks.jsonl.

This is the dataset->chunks half of indexing; build_index.py consumes the
output of this script (or rebuilds everything itself).

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --strategy parent_child --out data/processed/chunks.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking.factory import all_strategies  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.datasets.factory import get_dataset_loader  # noqa: E402
from app.indexing.build import chunk_documents, retrievable  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=all_strategies(),
        default=None,
        help="Chunking strategy (default: CHUNKING_STRATEGY from .env)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=get_settings().PROCESSED_CHUNKS_FILE,
        help="Output JSONL path for chunks",
    )
    args = parser.parse_args()

    settings = get_settings()
    strategy = args.strategy or settings.CHUNKING_STRATEGY

    documents = get_dataset_loader(settings).load_all()

    start = time.perf_counter()
    chunks = chunk_documents(documents, strategy)
    duration_ms = (time.perf_counter() - start) * 1000

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")

    print(f"Strategy:      {strategy}")
    print(f"Documents:     {len(documents)}")
    print(f"Chunks:        {len(chunks)} ({len(retrievable(chunks))} retrievable)")
    print(f"Chunking time: {duration_ms:.1f} ms")
    print(f"Wrote:         {out_path}")

    strategies = sorted({c.strategy for c in chunks})
    print(f"Strategies present: {json.dumps(strategies)}")


if __name__ == "__main__":
    main()
