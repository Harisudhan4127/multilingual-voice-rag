"""
Chunking strategy evaluation (Section 8).

For each chunking strategy: chunk the synthetic dataset, build a BM25 index
over the resulting chunks, run the hand-labeled eval queries (see
app/datasets/eval_queries.py), and measure:
  - retrieval recall@5 (did a chunk from the correct document appear in top 5?)
  - MRR (mean reciprocal rank of the first correct chunk)
  - latency (mean ms per query)
  - number of chunks produced
  - approximate index size (sum of chunk text length, as a size proxy)

All numbers are measured, not fabricated. Uses BM25-only retrieval since
embeddings/dense search are not wired in until Phase 3; this is documented
in the report's `notes` field rather than silently presented as a full
hybrid-retrieval evaluation.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --out benchmarks/chunking_results.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking.factory import all_strategies, get_chunker  # noqa: E402
from app.datasets.eval_queries import EVAL_QUERIES  # noqa: E402
from app.datasets.synthetic import SyntheticDatasetLoader  # noqa: E402
from app.indexing.bm25_store import BM25Store  # noqa: E402


def evaluate_strategy(strategy: str, documents, top_k: int = 5) -> dict:
    chunker = get_chunker(strategy)

    chunk_start = time.perf_counter()
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunker.chunk(doc))
    chunking_time_ms = (time.perf_counter() - chunk_start) * 1000

    # For parent_child, only child chunks are meaningful retrieval units
    # (parents exist purely for context expansion, not for standalone matching).
    if strategy == "parent_child":
        retrievable_chunks = [c for c in all_chunks if not c.is_parent]
    else:
        retrievable_chunks = all_chunks

    store = BM25Store()
    store.build(retrievable_chunks)

    reciprocal_ranks = []
    hits_at_5 = 0
    latencies_ms = []

    for eq in EVAL_QUERIES:
        start = time.perf_counter()
        results = store.search(eq.query, top_k=top_k)
        latencies_ms.append((time.perf_counter() - start) * 1000)

        relevant_docs = set(eq.relevant_document_ids)
        rank = None
        for i, (chunk, _score) in enumerate(results, start=1):
            if chunk.document_id in relevant_docs:
                rank = i
                break

        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            hits_at_5 += 1
        else:
            reciprocal_ranks.append(0.0)

    recall_at_5 = hits_at_5 / len(EVAL_QUERIES)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
    index_size_chars = sum(len(c.text) for c in retrievable_chunks)

    return {
        "strategy": strategy,
        "num_chunks": len(all_chunks),
        "num_retrievable_chunks": len(retrievable_chunks),
        "chunking_time_ms": round(chunking_time_ms, 3),
        "recall_at_5": round(recall_at_5, 4),
        "mrr": round(mrr, 4),
        "avg_query_latency_ms": round(avg_latency_ms, 4),
        "index_size_chars": index_size_chars,
        "num_eval_queries": len(EVAL_QUERIES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="benchmarks/chunking_results.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    documents = SyntheticDatasetLoader().load_all()
    print(f"Loaded {len(documents)} documents, {len(EVAL_QUERIES)} eval queries")

    results = []
    for strategy in all_strategies():
        print(f"Evaluating strategy: {strategy} ...")
        result = evaluate_strategy(strategy, documents, top_k=args.top_k)
        results.append(result)
        print(
            f"  chunks={result['num_chunks']} "
            f"recall@{args.top_k}={result['recall_at_5']} "
            f"mrr={result['mrr']} "
            f"latency={result['avg_query_latency_ms']}ms"
        )

    report = {
        "dataset": "synthetic",
        "num_documents": len(documents),
        "num_eval_queries": len(EVAL_QUERIES),
        "retrieval_method": "bm25_only",
        "notes": (
            "Retrieval signal is BM25-only. Dense/embedding-based retrieval is "
            "added in Phase 3-4; this evaluation will be re-run with hybrid "
            "dense+BM25 retrieval once available, and results updated -- not "
            "estimated in advance."
        ),
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {out_path}")


if __name__ == "__main__":
    main()
