"""
Latency benchmark (Section 24-25).

Runs N real queries through the REAL pipeline (configured embedder, Qdrant,
reranker, mock LLM/STT unless overridden via .env) and measures:
  - RAG latency:      query -> retrieval -> rerank -> generation -> guardrails
  - Full voice latency: audio -> STT -> RAG -> final response (mock STT adds
    only local processing time; with a hosted STT the network RTT dominates)
  - per-stage latencies

Everything is measured live. Nothing is fabricated. The response cache is
disabled during the run so numbers reflect cold-path behaviour.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --num-queries 200
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.datasets.factory import get_dataset_loader  # noqa: E402
from app.dependencies import build_pipeline  # noqa: E402
from app.utils.timing import latency_summary  # noqa: E402


def build_query_set(num_queries: int) -> list[str]:
    """
    In-distribution query set: the hand-labeled eval questions plus templated
    variations over corpus titles. Documented in the report so nobody mistakes
    them for an independent benchmark suite.
    """
    from app.datasets.eval_queries import EVAL_QUERIES

    documents = get_dataset_loader(get_settings()).load_all()
    titles = [doc.title for doc in documents]

    templates = [
        "What is {title}?",
        "Explain {title}.",
        "Tell me about {title}",
        "{title}",
    ]

    queries: list[str] = [eq.query for eq in EVAL_QUERIES]
    i = 0
    while len(queries) < num_queries:
        title = titles[i % len(titles)]
        queries.append(templates[(i // len(titles)) % len(templates)].format(title=title))
        i += 1
    return queries[:num_queries]


def percentile_stage(stage_values: dict[str, list[float]]) -> dict:
    return {stage: latency_summary(vals) for stage, vals in stage_values.items()}


async def run_benchmark(num_queries: int, include_voice: bool) -> dict:
    from app.datasets.eval_queries import EVAL_QUERIES

    settings = get_settings()
    components = await build_pipeline(settings)

    # Cold-path measurement: disable the response cache for this run.
    components.orchestrator.response_cache.ttl_seconds = -1.0

    queries = build_query_set(num_queries)
    print(f"Running {len(queries)} queries | reranker={components.reranker.kind} "
          f"embedder={components.embedder.kind} vector={components.vector_store.mode}")

    rag_latencies: list[float] = []
    voice_latencies: list[float] = []
    stages: dict[str, list[float]] = {}
    status_counts: dict[str, int] = {}
    warmup_done = False

    try:
        for idx, q in enumerate(queries):
            if not warmup_done:
                # Warmup pass (model load / first inference) excluded from stats.
                await components.orchestrator.run(q)
                warmup_done = True

            response = await components.orchestrator.run(q)
            rag_latencies.append(response.latency_ms)
            for key, value in response.timings.items():
                stages.setdefault(key, []).append(value)
            status_counts[response.status] = status_counts.get(response.status, 0) + 1
            if (idx + 1) % 25 == 0:
                print(f"  {idx + 1}/{len(queries)} done "
                      f"(p50 so far: {latency_summary(rag_latencies)['p50']} ms)")

        # --- Voice path: STT + full pipeline --------------------------------
        if include_voice:
            fake_audio = b"RIFF" + bytes(2048)  # ~128 ms of "audio" at 16 kB/s
            stt = components.stt_provider
            for _ in range(min(num_queries, 100)):
                t0 = time.perf_counter()
                transcript = await stt.transcribe(fake_audio)
                _ = await components.orchestrator.run(transcript.text)
                voice_latencies.append((time.perf_counter() - t0) * 1000)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "app_env": settings.APP_ENV,
            "dataset_provider": settings.DATASET_PROVIDER,
            "chunking_strategy": settings.CHUNKING_STRATEGY,
            "num_documents": len(get_dataset_loader(settings).load_all()),
            "num_queries": len(queries),
            "query_derivation": (
                "hand-labeled eval questions + templated title variants "
                "(in-distribution); not an independent benchmark suite"
            ),
            "warmup": "one untimed warmup query before measurement",
            "cache": "response cache disabled during measurement",
            "components": {
                "embedder": f"{components.embedder.kind}:{components.embedder.model_name}",
                "vector_store": components.vector_store.mode,
                "bm25_chunks": len(components.bm25_store),
                "reranker": components.reranker.kind,
                "llm": components.llm_provider.name,
                "stt": components.stt_provider.name,
            },
            "rag_latency_ms": latency_summary(rag_latencies),
            "rag_stage_latency_ms": percentile_stage(stages),
            "status_counts": status_counts,
            "notes": [
                "All values are measured live by scripts/benchmark.py; none are fabricated.",
                "LLM is the deterministic extractive mock; a hosted LLM would add network RTT.",
                "Voice latency uses MockSTTProvider (local); Sarvam STT would add network RTT.",
            ],
        }
        if voice_latencies:
            report["full_voice_latency_ms"] = latency_summary(voice_latencies)
        return report
    finally:
        await components.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-queries", type=int, default=100,
        help="Number of benchmark queries (default: 100)",
    )
    parser.add_argument(
        "--no-voice", action="store_true",
        help="Skip the full-voice (STT + RAG) measurement",
    )
    parser.add_argument(
        "--out", type=str, default="benchmarks/latency.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    report = asyncio.run(run_benchmark(args.num_queries, not args.no_voice))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = out_path.with_suffix(".md")
    write_markdown_report(report, md_path)

    print("\n--- Latency summary (ms) ---")
    for name in ("rag_latency_ms", "full_voice_latency_ms"):
        if name in report:
            print(f"{name}: {json.dumps(report[name])}")
    print(f"\nWrote {out_path}")
    print(f"Wrote {md_path}")


def write_markdown_report(report: dict, path: Path) -> None:
    def row(name: str, d: dict) -> str:
        return (f"| {name} | {d['min']} | {d['avg']} | {d['p50']} | {d['p70']} "
                f"| {d['p95']} | {d['p100']} |")

    lines = [
        "# Latency Benchmark Report",
        "",
        f"Generated: `{report['generated_at']}` · queries: **{report['num_queries']}** "
        f"· dataset: `{report['dataset_provider']}` ({report['num_documents']} docs) "
        f"· chunking: `{report['chunking_strategy']}`",
        "",
        f"Components: `{report['components']}`",
        "",
        "## RAG latency (ms)",
        "",
        "| metric | min | avg | P50 | P70 | P95 | P100 |",
        "|---|---|---|---|---|---|---|",
        row("RAG end-to-end", report["rag_latency_ms"]),
    ]
    for stage, summary in report.get("rag_stage_latency_ms", {}).items():
        lines.append(row(stage.replace("_ms", ""), summary))

    if "full_voice_latency_ms" in report:
        lines += [
            "",
            "## Full voice latency (ms) — audio → STT → RAG → response",
            "",
            "| metric | min | avg | P50 | P70 | P95 | P100 |",
            "|---|---|---|---|---|---|---|",
            row("voice end-to-end", report["full_voice_latency_ms"]),
        ]

    lines += ["", "## Response statuses", ""]
    for status, count in report.get("status_counts", {}).items():
        lines.append(f"- `{status}`: {count}")

    lines += ["", "## Notes", ""]
    lines += [f"- {n}" for n in report.get("notes", [])]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
