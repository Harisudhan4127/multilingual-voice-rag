# Latency Benchmark Report

Generated: `2026-08-22T09:11:02.059146+00:00` · queries: **100** · dataset: `synthetic` (17 docs) · chunking: `sentence`

Components: `{'embedder': 'sentence_transformers:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'vector_store': 'local', 'bm25_chunks': 30, 'reranker': 'cross_encoder', 'llm': 'mock', 'stt': 'mock'}`

## RAG latency (ms)

| metric | min | avg | P50 | P70 | P95 | P100 |
|---|---|---|---|---|---|---|
| RAG end-to-end | 1134.99 | 1682.63 | 1690.46 | 1818.54 | 2081.8 | 2462.38 |
| query_processing | 0.03 | 0.05 | 0.05 | 0.06 | 0.09 | 0.14 |
| input_guardrails | 0.02 | 0.04 | 0.03 | 0.04 | 0.07 | 1.02 |
| retrieval | 1.78 | 46.76 | 48.38 | 55.64 | 79.74 | 131.24 |
| rerank | 1077.06 | 1634.74 | 1634.86 | 1787.05 | 2005.99 | 2389.27 |
| context_selection | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.04 |
| generation | 0.29 | 0.5 | 0.46 | 0.52 | 0.84 | 1.15 |
| output_guardrails | 0.16 | 0.33 | 0.27 | 0.32 | 0.64 | 1.53 |
| total | 1134.99 | 1682.63 | 1690.46 | 1818.54 | 2081.8 | 2462.38 |

## Full voice latency (ms) — audio → STT → RAG → response

| metric | min | avg | P50 | P70 | P95 | P100 |
|---|---|---|---|---|---|---|
| voice end-to-end | 1212.9 | 1770.18 | 1729.36 | 1898.4 | 2332.33 | 2934.22 |

## Response statuses

- `ok`: 100

## Notes

- All values are measured live by scripts/benchmark.py; none are fabricated.
- LLM is the deterministic extractive mock; a hosted LLM would add network RTT.
- Voice latency uses MockSTTProvider (local); Sarvam STT would add network RTT.
