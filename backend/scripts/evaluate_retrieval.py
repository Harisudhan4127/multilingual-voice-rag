"""
Retrieval evaluation (Section 27): Recall@1/5/10 + MRR over the hand-labeled
EVAL_QUERIES, comparing four configurations:

    dense-only | bm25-only | hybrid (RRF) | hybrid + rerank

Uses the REAL configured embedder/reranker and Qdrant index built by
scripts/build_index.py. All numbers are measured live.

Usage:
    python scripts/build_index.py   # once
    python scripts/evaluate_retrieval.py
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.datasets.eval_queries import EVAL_QUERIES  # noqa: E402
from app.dependencies import build_pipeline  # noqa: E402
from app.indexing.build import load_chunks, retrievable  # noqa: E402
from app.utils.timing import latency_summary  # noqa: E402


def rank_metrics(results: list, relevant_ids: set[str], k: int) -> tuple[float, float]:
    """Return (hit@k, reciprocal_rank) for one ranked result list."""
    for rank, chunk in enumerate(results[:k], start=1):
        if chunk.document_id in relevant_ids:
            return 1.0, 1.0 / rank
    return 0.0, 0.0


async def evaluate(k_values=(1, 5, 10)) -> dict:
    settings = get_settings()
    components = await build_pipeline(settings)

    try:
        chunks = retrievable(load_chunks(settings.PROCESSED_CHUNKS_FILE))
        configs = {
            "dense": [],
            "bm25": [],
            "hybrid_rrf": [],
            "hybrid_rerank": [],
        }
        per_config: dict[str, dict] = {
            name: {f"recall@{k}": [] for k in k_values} | {"mrr": [], "latency_ms": []}
            for name in configs
        }

        print(f"Evaluating {len(EVAL_QUERIES)} eval queries "
              f"(embedder={components.embedder.kind}, reranker={components.reranker.kind})")

        for eq in EVAL_QUERIES:
            relevant = set(eq.relevant_document_ids)
            t0 = time.perf_counter()
            dense = await components.orchestrator.retriever.dense.search(
                eq.query, top_k=10
            )
            bm25 = await components.orchestrator.retriever.bm25.search(eq.query, top_k=10)
            base_ms = (time.perf_counter() - t0) * 1000

            from app.pipeline.hybrid_search import rrf_fuse

            fused = rrf_fuse(
                dense, bm25,
                rrf_k=settings.RRF_K,
                dense_weight=settings.DENSE_WEIGHT,
                bm25_weight=settings.BM25_WEIGHT,
                top_k=10,
            )
            t1 = time.perf_counter()
            reranked = await components.reranker.rerank(
                eq.query, fused, top_k=settings.TOP_K_RERANK
            )
            rerank_ms = (time.perf_counter() - t1) * 1000

            ranked = {
                "dense": [c for c, _ in dense],
                "bm25": [c for c, _ in bm25],
                "hybrid_rrf": [c for c, _ in fused],
                "hybrid_rerank": [c for c, _ in reranked],
            }
            latencies = {
                "dense": base_ms,
                "bm25": base_ms,
                "hybrid_rrf": base_ms,
                "hybrid_rerank": base_ms + rerank_ms,
            }
            # MRR computed against recall@10 lists.
            for name, chunk_list in ranked.items():
                for k in k_values:
                    hit, _ = rank_metrics(chunk_list, relevant, k)
                    per_config[name][f"recall@{k}"].append(hit)
                _, rr = rank_metrics(chunk_list, relevant, max(k_values))
                per_config[name]["mrr"].append(rr)
                per_config[name]["latency_ms"].append(latencies[name])

        def summarize(values: list[float]) -> float:
            return round(sum(values) / len(values), 4) if values else 0.0

        summary = {
            name: {
                metric: (
                    latency_summary(vals)
                    if metric == "latency_ms"
                    else summarize(vals)
                )
                for metric, vals in metrics.items()
            }
            for name, metrics in per_config.items()
        }

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "num_eval_queries": len(EVAL_QUERIES),
            "num_indexed_chunks": len(chunks),
            "components": {
                "embedder": f"{components.embedder.kind}:{components.embedder.model_name}",
                "reranker": components.reranker.kind,
                "vector_store": components.vector_store.mode,
            },
            "results": summary,
            "notes": [
                "Relevance labels come from app/datasets/eval_queries.py "
                "(hand-labeled against the synthetic dataset).",
                "A chunk counts as correct if its document_id is in the labeled set.",
                "All values are measured live by scripts/evaluate_retrieval.py.",
            ],
        }
        return report
    finally:
        await components.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=str, default="benchmarks/retrieval_eval.json"
    )
    args = parser.parse_args()

    report = asyncio.run(evaluate())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'config':16s} {'R@1':>7s} {'R@5':>7s} {'R@10':>7s} {'MRR':>7s}")
    for name, metrics in report["results"].items():
        print(
            f"{name:16s} "
            f"{metrics['recall@1']:7.3f} "
            f"{metrics['recall@5']:7.3f} "
            f"{metrics['recall@10']:7.3f} "
            f"{metrics['mrr']:7.3f}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
