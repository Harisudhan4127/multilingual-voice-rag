# HH Goa 2026 — Master Coding Prompt

You are a senior AI/ML + full-stack engineer. Build a complete, production-structured **Voice-Enabled RAG system** for the HH Goa 2026 Shortlisting Task 2.

## 1. Core Objective

Build this pipeline:

**Voice Input → Speech-to-Text → Query Processing → Hybrid Retrieval → Reranking → Context Selection → LLM Answer → Grounding/Safety Guardrails → Structured Response**

Target dataset:

**AI4Bharat MSMARCO-XI**

Dataset URL:

https://huggingface.co/datasets/ai4bharat/MSMARCO-XI

However, development must work **offline/local first**.

Do NOT block development because Hugging Face, Sarvam, hosted Qdrant, or a hosted LLM may be unavailable.

---

# 2. Development Strategy

Build the system in stages.

### Stage 1 — Fully local working MVP

Use:

- Synthetic MSMARCO-XI-like dataset
- Local embedding model
- Local Qdrant
- Local BM25
- Local/lightweight reranker
- Mock STT
- Mock LLM
- FastAPI
- React + Vite

The complete application must run locally.

### Stage 2 — Real dataset adapter

Add support for the real MSMARCO-XI dataset without changing the RAG architecture.

### Stage 3 — Real provider adapters

Add:

- Sarvam STT provider
- Real LLM provider

All external providers must be behind clean interfaces.

Never hardcode provider-specific logic throughout the application.

---

# 3. Non-Negotiable Architecture

Use provider interfaces.

Example:

```text
STTProvider
├── MockSTTProvider
└── SarvamSTTProvider

LLMProvider
├── MockLLMProvider
└── RealLLMProvider

VectorStore
└── QdrantVectorStore
```

Configuration must determine which implementation is used.

Example:

```env
STT_PROVIDER=mock
LLM_PROVIDER=mock
VECTOR_PROVIDER=qdrant
```

Later:

```env
STT_PROVIDER=sarvam
LLM_PROVIDER=real
```

No major code rewrite should be required.

---

# 4. Technology Stack

## Backend

- Python 3.11+
- FastAPI
- Pydantic
- Uvicorn
- Qdrant
- sentence-transformers
- BM25 implementation
- NumPy
- httpx
- pytest

## Frontend

- React
- Vite
- JavaScript/TypeScript
- Modern CSS

Do not introduce unnecessary frameworks.

## Infrastructure

- Docker
- Docker Compose

Redis may be added only if useful. Prefer an in-memory cache initially to reduce complexity.

---

# 5. Project Structure

Create:

```text
voice-rag/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── dependencies.py
│   │   │
│   │   ├── api/
│   │   │   ├── voice.py
│   │   │   ├── query.py
│   │   │   ├── health.py
│   │   │   └── metrics.py
│   │   │
│   │   ├── pipeline/
│   │   │   ├── orchestrator.py
│   │   │   ├── query_processor.py
│   │   │   ├── retriever.py
│   │   │   ├── hybrid_search.py
│   │   │   ├── reranker.py
│   │   │   ├── generator.py
│   │   │   └── guardrails.py
│   │   │
│   │   ├── chunking/
│   │   │   ├── base.py
│   │   │   ├── sentence.py
│   │   │   ├── token.py
│   │   │   ├── semantic.py
│   │   │   └── parent_child.py
│   │   │
│   │   ├── indexing/
│   │   │   ├── embeddings.py
│   │   │   ├── qdrant_store.py
│   │   │   └── bm25_store.py
│   │   │
│   │   ├── providers/
│   │   │   ├── stt/
│   │   │   │   ├── base.py
│   │   │   │   ├── mock.py
│   │   │   │   └── sarvam.py
│   │   │   │
│   │   │   └── llm/
│   │   │       ├── base.py
│   │   │       ├── mock.py
│   │   │       └── real.py
│   │   │
│   │   ├── schemas/
│   │   │   └── models.py
│   │   │
│   │   └── utils/
│   │       ├── cache.py
│   │       ├── logging.py
│   │       └── timing.py
│   │
│   ├── scripts/
│   │   ├── create_sample_dataset.py
│   │   ├── preprocess.py
│   │   ├── build_index.py
│   │   └── benchmark.py
│   │
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── benchmarks/
│
├── docker/
│   └── docker-compose.yml
│
├── README.md
├── .gitignore
└── docker-compose.yml
```

Do not create unnecessary files.

---

# 6. Dataset Layer

Initially generate a synthetic dataset resembling MSMARCO-style passages.

It should contain:

- document ID
- passage ID
- title
- text
- language
- metadata

Generate enough records to meaningfully test retrieval.

Create a script:

```bash
python scripts/create_sample_dataset.py
```

The real dataset must later be loadable through a separate dataset adapter.

Do NOT hardcode synthetic data directly inside application logic.

---

# 7. Chunking

Implement multiple chunking strategies.

Required:

1. Sentence-based
2. Token-based
3. Semantic
4. Parent-child

Every chunk should contain metadata:

```json
{
  "document_id": "...",
  "chunk_id": "...",
  "strategy": "...",
  "position": 0,
  "total_chunks": 5,
  "language": "...",
  "title": "...",
  "text": "..."
}
```

Create a common interface:

```python
class Chunker:
    def chunk(self, document):
        ...
```

Do not implement one fixed-size splitter and call it "advanced chunking."

---

# 8. Chunking Evaluation

Create an offline evaluation comparing strategies.

Measure:

- retrieval recall
- MRR
- latency
- number of chunks
- index size

Output a JSON report.

Example:

```text
benchmarks/chunking_results.json
```

Do not fabricate metrics.

---

# 9. Embeddings

Use a local sentence-transformers embedding model.

The model should be configurable through `.env`.

Example:

```env
EMBEDDING_MODEL=...
```

Load the model once during application startup.

Never load the embedding model per request.

If the preferred model is unavailable locally, provide a configurable fallback and clearly document it.

---

# 10. Vector Database

Use local Qdrant.

Docker:

```text
qdrant/qdrant
```

Create a reusable `QdrantVectorStore`.

It must support:

- create collection
- insert/upsert
- search
- metadata filtering
- health check

Do not use hosted Qdrant for the default local configuration.

---

# 11. BM25

Implement a local BM25 index.

It should support:

```python
search(query, top_k)
```

Store document/chunk IDs with results.

---

# 12. Hybrid Retrieval

Run:

```text
Dense Retrieval
+
BM25
↓
RRF Fusion
↓
Top-K candidates
```

Prefer parallel execution.

Use Reciprocal Rank Fusion rather than simply concatenating lists.

Make fusion configurable.

Example:

```env
DENSE_WEIGHT=0.65
BM25_WEIGHT=0.35
```

---

# 13. Reranking

Implement a reranking stage after hybrid retrieval.

Flow:

```text
Dense Top 20
BM25 Top 20
↓
Fusion
↓
Top 10–20
↓
Reranker
↓
Top 5
```

Use a lightweight local model where practical.

Do not use a huge model that makes the benchmark meaningless.

---

# 14. Query Processing

Create a query processor that performs:

- whitespace normalization
- speech artifact cleanup
- language detection where practical
- basic query classification
- empty-query validation

Do not aggressively rewrite user questions.

---

# 15. STT Interface

Create:

```python
class STTProvider:
    async def transcribe(self, audio):
        ...
```

Implement:

### MockSTTProvider

Used by default.

It should allow testing without external APIs.

### SarvamSTTProvider

Implement the real API adapter cleanly.

All API credentials must come from environment variables.

Never hardcode API keys.

---

# 16. LLM Interface

Create:

```python
class LLMProvider:
    async def generate(self, query, context):
        ...
```

Implement:

### MockLLMProvider

It should generate deterministic answers from retrieved context so the complete system can be tested offline.

### RealLLMProvider

Use an external API through a configurable endpoint/client.

Do not couple the rest of the application to one specific LLM provider.

---

# 17. Structured LLM Output

The answer generator should return structured data:

```json
{
  "answer": "...",
  "citations": [],
  "confidence": 0.0
}
```

Validate it with Pydantic.

Never blindly trust raw LLM output.

If structured output is invalid:

1. Retry once if appropriate.
2. Otherwise return a controlled error.

---

# 18. RAG Prompt

The LLM must be instructed:

```text
Answer ONLY using the supplied context.

If the context does not contain sufficient evidence,
do not invent an answer.

State that the information is insufficient.

Return citations for factual claims.
```

Never allow the model to treat its general knowledge as the primary source.

---

# 19. Guardrails

Implement at least:

### Guardrail A — Off-topic

Reject questions that cannot reasonably be answered from the dataset.

### Guardrail B — Unsafe

Reject unsafe/inappropriate requests.

### Guardrail C — Retrieval confidence

If retrieval confidence is too low, refuse.

### Guardrail D — Grounding

Check whether generated claims are supported by retrieved context.

### Guardrail E — Citation validation

Ensure citations correspond to actually retrieved chunks.

### Guardrail F — Hallucination prevention

If grounding fails:

```text
I don't have enough information in the provided dataset to answer this reliably.
```

Do not invent citations.

---

# 20. Harness / Orchestrator

The main request must pass through a structured orchestrator.

Example:

```text
Request
 ↓
Validate
 ↓
STT
 ↓
Query Processing
 ↓
Guardrail
 ↓
Dense + BM25
 ↓
Fusion
 ↓
Reranker
 ↓
Context Selection
 ↓
LLM
 ↓
Grounding
 ↓
Final Guardrail
 ↓
Response
```

Every stage should have:

- timeout
- exception handling
- structured result
- timing
- logging
- controlled retry where appropriate

Do not implement the whole pipeline as one giant function.

---

# 21. Request State

Use a structured state object.

Example:

```python
class PipelineState:
    request_id: str
    transcript: str | None
    query: str
    retrieval_results: list
    reranked_results: list
    context: list
    answer: str | None
    citations: list
    confidence: float
    grounded: bool
    timings: dict
    status: str
```

---

# 22. API Endpoints

Implement:

```text
GET  /api/v1/health
POST /api/v1/query
POST /api/v1/voice
GET  /api/v1/metrics
```

Example `/query`:

```json
{
  "query": "What is ...?"
}
```

Response:

```json
{
  "request_id": "...",
  "answer": "...",
  "sources": [],
  "confidence": 0.92,
  "grounded": true,
  "latency_ms": 120,
  "timings": {}
}
```

---

# 23. Frontend

Create a simple professional React UI.

Required:

- microphone button
- recording state
- transcript display
- answer display
- source citations
- confidence
- grounded status
- latency
- error messages

Do NOT spend excessive time on visual design.

The demo must make the RAG pipeline obvious.

---

# 24. Benchmarking

Create:

```bash
python scripts/benchmark.py
```

Run against at least 100 test queries initially.

Make the number configurable.

Measure:

- minimum
- average
- P50
- P70
- P95
- P100
- retrieval latency
- reranking latency
- generation latency
- total latency

Save:

```text
benchmarks/latency.json
```

Also generate a human-readable report.

---

# 25. Critical Latency Rule

Never fabricate the 200 ms requirement.

Measure actual performance.

Report two metrics:

### RAG latency

```text
Query received
→ retrieval
→ reranking
→ generation
→ guardrails
```

### Full voice latency

```text
Audio
→ STT
→ RAG
→ final response
```

External STT/LLM network latency must not be hidden.

If the system does not meet 200 ms, report the actual result and identify the bottleneck.

---

# 26. Latency Optimization

Implement:

- model warmup
- persistent model instances
- persistent Qdrant connection
- cached embeddings
- query cache
- parallel dense/BM25 retrieval
- small retrieval candidate set
- lightweight reranker
- small final context
- async API operations where useful

Do not optimize prematurely before obtaining a baseline.

---

# 27. Evaluation

Create evaluation scripts for:

### Retrieval

- Recall@1
- Recall@5
- Recall@10
- MRR

### Answer

- correctness
- relevance
- grounding
- citation correctness
- refusal correctness

### Safety

Test:

1. normal query
2. off-topic query
3. unanswerable query
4. unsafe query
5. empty query
6. malformed request

---

# 28. Tests

Write tests for:

- chunking
- BM25
- hybrid fusion
- reranking
- guardrails
- citation validation
- mock STT
- mock LLM
- API health
- query endpoint
- error handling

The project should have a meaningful automated test suite.

---

# 29. Docker

Provide Docker Compose for:

```text
backend
frontend
qdrant
```

The default configuration must work locally.

Do not require external APIs for the basic demo.

---

# 30. Configuration

Create `.env.example`.

Include:

```env
APP_ENV=development

STT_PROVIDER=mock
LLM_PROVIDER=mock

SARVAM_API_KEY=

LLM_API_KEY=
LLM_BASE_URL=

QDRANT_URL=http://localhost:6333

EMBEDDING_MODEL=

TOP_K_DENSE=20
TOP_K_BM25=20
TOP_K_RERANK=5

DENSE_WEIGHT=0.65
BM25_WEIGHT=0.35

RETRIEVAL_CONFIDENCE_THRESHOLD=
GROUNDING_THRESHOLD=
```

Never commit `.env`.

---

# 31. Error Handling

The system must gracefully handle:

- no microphone input
- empty transcript
- STT failure
- vector DB unavailable
- embedding failure
- BM25 failure
- LLM timeout
- invalid LLM response
- no relevant documents
- grounding failure
- malformed request

Never expose stack traces to the frontend in production mode.

---

# 32. README

Create a comprehensive but concise README containing:

1. Project overview
2. Architecture
3. Dataset
4. Chunking strategies
5. Hybrid retrieval
6. Reranking
7. Voice architecture
8. Guardrails
9. Installation
10. Local setup
11. Docker setup
12. Environment variables
13. Dataset preparation
14. Index building
15. Running backend
16. Running frontend
17. Benchmarking
18. Evaluation
19. Latency results
20. Switching mock providers to real providers
21. Limitations
22. Future work

Do not put fake benchmark numbers in README.

Use placeholders such as:

```text
P50: generated by benchmark
P70: generated by benchmark
P100: generated by benchmark
```

until the benchmark is actually executed.

---

# 33. Coding Rules

Follow these rules strictly:

- Write clean modular Python.
- Use type hints.
- Use Pydantic models.
- Use dependency injection where useful.
- Avoid global mutable state.
- Avoid unnecessary abstractions.
- Do not duplicate code.
- Keep provider implementations isolated.
- Keep configuration centralized.
- Never hardcode secrets.
- Never fabricate evaluation results.
- Never silently swallow exceptions.
- Log useful errors.
- Prefer simple implementations over unnecessary frameworks.

---

# 34. Token/Implementation Efficiency

You are working as a coding agent.

Do NOT dump thousands of lines of code into one response before implementing.

Work incrementally.

For each stage:

1. Inspect the current project.
2. Implement only that stage.
3. Run/test it.
4. Fix errors.
5. Continue to the next stage.

Reuse existing files.

Do not rewrite working code unnecessarily.

Do not create duplicate implementations.

If a dependency is unavailable, create a clean fallback rather than blocking the entire project.

---

# 35. Implementation Order

Follow exactly this order:

```text
PHASE 1
Project structure
Configuration
FastAPI health endpoint
React frontend
Docker

PHASE 2
Synthetic dataset
Dataset loader
Chunking strategies
Chunking evaluation

PHASE 3
Embeddings
Qdrant
BM25
Index builder

PHASE 4
Dense retrieval
BM25 retrieval
RRF hybrid retrieval
Reranking

PHASE 5
Mock LLM
RAG generation
Structured output

PHASE 6
Guardrails
Grounding
Citations
Refusal logic

PHASE 7
Pipeline orchestrator
Retries
Timeouts
Error recovery

PHASE 8
Mock STT
Voice endpoint
React microphone UI

PHASE 9
Benchmarking
Latency metrics
Retrieval evaluation

PHASE 10
Sarvam adapter
Real LLM adapter
Real MSMARCO-XI adapter

PHASE 11
Docker integration
Final tests
README
```

---

# 36. Important Constraint

The project must remain runnable in **offline/local mode** even after real integrations are added.

Default:

```text
STT = Mock
LLM = Mock
Dataset = Synthetic
Vector DB = Local Qdrant
```

Production/test mode:

```text
STT = Sarvam
LLM = Real
Dataset = MSMARCO-XI
Vector DB = Local Qdrant or configured provider
```

---

# 37. Final Acceptance Criteria

The implementation is complete only when:

- [ ] Full repository exists
- [ ] Backend starts
- [ ] Frontend starts
- [ ] Qdrant starts
- [ ] Synthetic dataset loads
- [ ] Index can be built
- [ ] Dense retrieval works
- [ ] BM25 works
- [ ] Hybrid retrieval works
- [ ] Reranking works
- [ ] Mock LLM works
- [ ] Guardrails work
- [ ] Grounding validation works
- [ ] Query API works
- [ ] Mock voice API works
- [ ] Frontend voice flow works
- [ ] Benchmark executes
- [ ] Latency metrics are generated
- [ ] Automated tests pass
- [ ] Docker Compose works
- [ ] Real provider adapters exist
- [ ] MSMARCO-XI adapter exists
- [ ] README is complete
- [ ] No fake benchmark claims exist
- [ ] No secrets are committed

---

# 38. First Action

Do NOT attempt to implement the entire project in one step.

Start with **PHASE 1 only**.

First inspect the existing directory and determine whether a project already exists.

Then create the minimal working:

```text
backend/
frontend/
docker-compose.yml
.env.example
.gitignore
README.md
```

Make sure:

```text
FastAPI → /api/v1/health
React → working page
Qdrant → reachable
Docker Compose → starts
```

Test Phase 1 before proceeding.

After Phase 1 is working, continue automatically to Phase 2.

At every phase, preserve working functionality from previous phases.

The ultimate goal is a **real, demonstrable, modular HH Goa 2026 Voice-Enabled RAG system**, not merely a collection of code stubs.