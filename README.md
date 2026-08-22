# Voice-Enabled RAG — HH Goa 2026 (Shortlisting Task 2)

A voice-enabled Retrieval-Augmented Generation system, built local/offline-first
and upgraded to real providers (Sarvam STT, hosted LLM, the real MSMARCO-XI
dataset) through **configuration only** — no architecture rewrites.

**Status: all 11 phases complete. 115 tests passing. All benchmark numbers in
this README are measured by scripts in `backend/scripts/`, none fabricated.**

## Pipeline

```
Voice Input → STT → Query Processing → Hybrid Retrieval (dense + BM25 + RRF)
→ Reranking → Context Selection → LLM Generation → Grounding/Safety Guardrails
→ Structured Response (+ citations)
```

Text queries skip STT: `POST /api/v1/query` runs everything from query
processing onward.

## Architecture

Every external dependency sits behind a small interface, and a factory is the
only place that branches on configuration:

| Interface | Implementations | Selected by |
|---|---|---|
| `STTProvider` | `Mock`, `Sarvam` | `STT_PROVIDER` |
| `DatasetLoader` | `Synthetic`, `MSMarcoXI` | `DATASET_PROVIDER` |
| `Chunker` | sentence / token / semantic / parent-child | `CHUNKING_STRATEGY` |
| `Embedder` | SentenceTransformer, Hashing fallback | `EMBEDDING_MODEL` |
| `VectorStore` | Qdrant (server → embedded-local fallback) | `QDRANT_URL` |
| `Reranker` | CrossEncoder, Lexical fallback | `RERANKER_MODEL` |
| `LLMProvider` | Mock, Real (OpenAI-compatible) | `LLM_PROVIDER` |

Backend layout (`backend/app/`): `api/` (FastAPI routers), `pipeline/`
(orchestrator, retrievers, reranker, generator, guardrails, query processor),
`indexing/` (embedders, Qdrant store, index builder, BM25 store),
`chunking/`, `datasets/`, `providers/{stt,llm}`, `schemas/`, `utils/`
(TTL cache, timing, metrics).

The pipeline is orchestrated by `PipelineOrchestrator`: per-stage timeouts,
bounded retries with backoff for retrieval/rerank/generation, graceful
degradation (reranker failure falls back to fused ranking; LLM failure after
retries returns a refusal rather than an error), per-stage latency captured in
every response.

## Dataset

- Default: synthetic MSMARCO-style dataset — 17 hand-written passages across 5
  topically distinct domains (`scripts/create_sample_dataset.py`).
- Real: `ai4bharat/MSMARCO-XI` loader (`app/datasets/msmarco_xi.py`) streams
  any of the 14 Indic-language configs from the HF Hub and maps rows into the
  same `Document` schema. Caps at `MSMARCO_MAX_DOCS` passages; selected
  passages sort first. Requires `pip install datasets`.

Chunking/indexing/retrieval are identical under either source.

## Chunking

Four strategies behind one `Chunker.chunk(document)` interface:
`sentence` (N sentences + overlap), `token` (token windows + overlap),
`semantic` (greedy grouping while consecutive-sentence similarity stays above
threshold; similarity function pluggable), `parent_child` (small precise child
chunks linked to large context-bearing parents via `parent_id`; retrieval hits
children, generation sees parents).

Measured comparison (BM25 retrieval, 17 labeled queries,
`scripts/evaluate_chunking.py` → `benchmarks/chunking_results.json`):

| Strategy | Chunks | Recall@5 | MRR | Chunking ms |
|---|---|---|---|---|
| sentence | 30 | 1.0 | 1.0 | 0.36 |
| token | 34 | 1.0 | 1.0 | 0.25 |
| semantic | 43 | 1.0 | 1.0 | 0.94 |
| parent_child | 54 (37 retrievable) | 1.0 | 1.0 | 0.53 |

**Honest caveat:** scores saturate because the corpus is tiny and topically
well-separated — this evaluation cannot rank strategies against each other.
`sentence` is default (best chunk count vs. quality balance here).

## Embeddings & Indexing

- Primary embedder: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim,
  multilingual — matches Indic scope). Hashing-based fallback if
  sentence-transformers is unavailable.
- Vector store: Qdrant. Tries `QDRANT_URL`, transparently falls back to
  embedded local mode persisted at `data/qdrant_local/` (zero-setup demo).
- BM25 index built over the same chunks (in-memory).
- One command builds both: `python scripts/build_index.py` (or automatic on
  API startup when the index is missing/stale and `AUTO_INDEX_ON_STARTUP=true`).

## Retrieval

1. **Query processing** — language detection (Devanagari/Bengali/etc. script
   ranges + keyword heuristics), normalization, safety pre-filter.
2. **Hybrid search** — dense (cosine) + sparse (BM25), fused with **RRF**
   (`k=60`, weights configurable via `HYBRID_DENSE_WEIGHT` /
   `HYBRID_BM25_WEIGHT`). Either leg's failure degrades to the survivor.
3. **Reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` over RRF's top-N;
   lexical F1 fallback reranker when the model isn't available.
4. **Confidence gate** — best score below `RETRIEVAL_CONFIDENCE_THRESHOLD`
   ⇒ refusal instead of a hallucinated answer.

## Generation & Guardrails

- `LLMProvider.generate(prompt)` returns raw text; the generator parses it
  into structured output `{answer, citations[], confidence}` with bounded
  JSON-repair retries, then validates every citation against actually
  retrieved chunks — invalid ones are dropped, not passed through.
- Mock provider is deterministic-extractive (top sentences from cited chunks)
  so demos/tests never depend on network or randomness.
- **Grounding**: retrieved-context overlap check; below threshold ⇒ canonical
  refusal (`REFUSAL_ANSWER`, defined once in `app/schemas/models.py`).
- **Safety**: input side blocks unsafe queries (`refused_unsafe`); output side
  scans answers before returning.
- **Off-topic**: vocabulary-overlap gate between query and corpus
  (`refused_off_topic`).
- **Citations** reference `chunk_id`s that exist in the response's `sources`.

## Voice

`POST /api/v1/voice` accepts multipart audio (browser records webm via
MediaRecorder). Audio → `STTProvider.transcribe()` → transcript flows through
the same orchestrator as text queries; transcript language feeds query
processing. Sarvam adapter posts multipart to
`${SARVAM_BASE_URL}/speech-to-text` (`saarika:v2`, key via header) — endpoint
path/model are configurable, credentials come only from env. Frontend
`MicButton` handles record/stop states and permission errors.

## API

Base URL `http://localhost:8000/api/v1`. Interactive docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + component status |
| POST | `/query` | text query → full RAG pipeline |
| POST | `/voice` | multipart audio → STT → full RAG pipeline |
| GET | `/metrics` | process/pipeline counters |

`POST /api/v1/query` example:

```json
// request
{"query": "What is supervised learning?"}
// response (abridged)
{
  "request_id": "44c0a308-…",
  "answer": "Supervised learning is a machine learning paradigm where …",
  "status": "ok",
  "confidence": 0.87,
  "grounded": true,
  "citations": ["doc_0001_sentence_child_0"],
  "sources": [{"chunk_id": "doc_0001_sentence_child_0", "title": "Introduction to Supervised Learning", "score": 0.91, "text": "…"}],
  "latency_ms": 1690.5,
  "timings": {"retrieval": 48.4, "rerank": 1634.9, "generation": 0.5}
}
```

Refusal statuses: `refused_unsafe`, `refused_off_topic`,
`refused_no_context`, `grounding_failed` — always HTTP 200 with the canonical
refusal answer; malformed input is HTTP 422.

## Installation

### Local

```bash
# backend
cd backend
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt      # CPU-only torch pinned
cp .env.example .env
./.venv/bin/uvicorn app.main:app --reload --port 8000
# first boot auto-builds the index (~10–20s model load)

# frontend
cd frontend
npm install
npm run dev        # http://localhost:5173, proxies /api -> :8000
```

### Docker

```bash
docker compose up --build
```

- `qdrant` (:6333) with persistent volume; `backend` (:8000) waits for its
  healthcheck; models are baked into the backend image so runtime needs no HF
  access; `frontend` (:5173 → nginx) serves the production build and reverse-
  proxies `/api` to the backend.
- `backend/.env` is optional — without it everything runs offline with mocks.

> Verified note: compose/Dockerfiles were written carefully but **not
> executed** — the development sandbox has no Docker daemon. Expect first-run
> friction only around image builds; the service wiring matches the verified
> local setup.

## Configuration

Full documented list in `backend/.env.example`. Key groups:

```env
STT_PROVIDER=mock            # sarvam + SARVAM_API_KEY for real
LLM_PROVIDER=mock            # real + LLM_BASE_URL/LLM_API_KEY/LLM_MODEL
DATASET_PROVIDER=synthetic   # msmarco_xi + DATASET_LANGUAGE/MSMARCO_SPLIT/MSMARCO_MAX_DOCS
CHUNKING_STRATEGY=sentence
QDRANT_URL=                  # empty = embedded local mode
RETRIEVAL_CONFIDENCE_THRESHOLD=0.15   # cross-encoder sigmoid scale
```

## Switching to Real Providers

```env
STT_PROVIDER=sarvam
SARVAM_API_KEY=svak_…

LLM_PROVIDER=real             # any OpenAI-compatible endpoint
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-…
LLM_MODEL=gpt-4o-mini

DATASET_PROVIDER=msmarco_xi   # pip install datasets
```

Then rebuild the index (`python scripts/build_index.py`) and restart. No code
changes — that is the point of the provider interfaces. The mock providers
remain available for offline development and CI.

## Benchmarks & Evaluation (measured)

Machine: CPU-only sandbox (no GPU), torch 2.13.0+cpu, Python 3.14 venv.
Raw artifacts: `backend/benchmarks/*.json` + `latency.md`.

### Latency (100 mixed queries, `scripts/benchmark.py`)

| Stage | avg ms | P50 | P95 | P100 |
|---|---|---|---|---|
| query_processing | 0.05 | 0.05 | 0.09 | 0.14 |
| retrieval (hybrid+RRF) | 46.8 | 48.4 | 79.7 | 131.2 |
| rerank (cross-encoder, CPU) | **1634.7** | 1634.9 | 2006.0 | 2389.3 |
| generation (mock LLM) | 0.5 | 0.5 | 0.8 | 1.2 |
| guardrails (in+out) | 0.37 | 0.31 | 0.71 | 1.53 |
| **RAG end-to-end** | **1682.6** | **1690.5** | **2081.8** | **2462.4** |
| voice end-to-end (mock STT) | 1770.2 | 1729.4 | 2332.3 | 2934.2 |

**Dominant cost is cross-encoder inference on CPU** (~97% of end-to-end);
retrieval adds <50ms. With a GPU/hosted reranker — or by lowering
`RERANK_TOP_K` — P95 drops proportionally. A real LLM/Sarvam add their network
RTT on top.

### Retrieval quality (17 labeled queries, `scripts/evaluate_retrieval.py`)

| Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| dense | 1.0 | 1.0 | 1.0 | 1.0 |
| bm25 | 1.0 | 1.0 | 1.0 | 1.0 |
| hybrid_rrf | 1.0 | 1.0 | 1.0 | 1.0 |
| hybrid_rerank | 1.0 | 1.0 | 1.0 | 1.0 |

**Honest caveat:** perfect scores reflect a tiny, topically well-separated
synthetic corpus — they validate *correctness of the plumbing* (all methods
find the right document; no integration bugs), not that hybrid > dense. On
MSMARCO-XI these numbers will differentiate.

### Answer quality & grounding (same 17 queries, `scripts/evaluate_answers.py`)

| Metric | Value |
|---|---|
| ok status rate | 1.00 |
| grounded rate | 1.00 |
| citations valid (exist in sources) | 1.00 |
| citations relevant (heuristic overlap) | 0.65 |
| answer support proxy | 0.79 |

Citation *validity* is perfect; heuristic *relevance* at 0.65 flags that the
extractive mock sometimes cites adjacent chunks — a real LLM plus stricter
citation validation is the intended fix.

### Safety behavior (`scripts/evaluate_safety.py` — live HTTP, real pipeline)

All 6 cases pass: normal→ok, off-topic→refused_off_topic, unanswerable→
refusal, unsafe→refused_unsafe, empty/malformed→HTTP 422.

## Testing

```bash
cd backend && ./.venv/bin/python -m pytest tests/ -v   # 115 passed
```

Coverage spans chunking strategies, embeddings (real + hashing fallback),
Qdrant store (local mode), hybrid fusion math, rerankers, generation contract,
guardrails, orchestrator retries/timeouts/degradation, API contracts, voice
endpoint, and all Phase-10 adapters against mocked HTTP (no external calls in
CI). Provider factories fail loudly on missing credentials/config instead of
misrouting to mocks.

## Limitations

- Cross-encoder latency on CPU dominates end-to-end time (see benchmarks).
- Evaluation corpus is small and synthetic; quality metrics saturate and will
  only become meaningful on MSMARCO-XI-scale data.
- `citations_relevant_rate` (0.65) shows the extractive mock cites some
  adjacent chunks; citation precision needs a real LLM.
- Docker files unexecuted in the build sandbox (no daemon) — see Installation.
- No auth/rate limiting; single-process, in-memory BM25/cache (by design for
  the task scope).
- Semantic chunking uses lexical similarity as its pluggable signal; swapping
  in embedding cosine is one function away but not benchmarked here.

## Future Work

GPU/hosted reranking + caching for latency; MSMARCO-XI end-to-end run with
per-language eval sets; embedding-cosine semantic chunking benchmark; streaming
responses (SSE); conversation memory; auth + rate limits; k8s manifests.

---

## Implementation Progress

- [x] Phase 1 — structure, config, FastAPI health + stub, React shell
- [x] Phase 2 — synthetic dataset, 4 chunkers, BM25, chunking eval
- [x] Phase 3 — embeddings (multilingual MiniLM + fallback), Qdrant
      (server/local), index builder
- [x] Phase 4 — dense + BM25 + RRF hybrid, cross-encoder reranking
- [x] Phase 5 — mock LLM, grounded structured generation, citations
- [x] Phase 6 — guardrails: grounding, safety, off-topic, refusals
- [x] Phase 7 — orchestrator: retries, timeouts, degradation, timings
- [x] Phase 8 — STT interface + mock, `/api/v1/voice`, mic UI
- [x] Phase 9 — latency benchmark, retrieval/answer/safety evaluations
- [x] Phase 10 — Sarvam STT, OpenAI-compatible real LLM, MSMARCO-XI loaders
- [x] Phase 11 — production Dockerfiles, compose healthchecks, README
