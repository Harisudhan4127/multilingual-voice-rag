import { useEffect, useState } from "react";
import MicButton from "./components/MicButton";
import { getHealth, postQuery, postVoice } from "./services/api";
import "./App.css";

// Phase 8 scope: full voice + text flow. The mic records -> /api/v1/voice ->
// transcript -> RAG pipeline -> grounded answer with sources. A dev-only
// "mock transcript" field drives the voice endpoint without a microphone
// (the backend honors it only when STT_PROVIDER=mock).
export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);

  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastTranscript, setLastTranscript] = useState(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((e) => setHealthError(e.message));
  }, []);

  async function applyResponse(promise) {
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      setResponse(await promise);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!query.trim()) return;
    setLastTranscript(null);
    applyResponse(postQuery(query));
  }

  function handleRecording(blob) {
    // Mock STT derives a deterministic canned question from the audio bytes,
    // so the recorded clip alone is enough to demo the voice loop offline.
    setLastTranscript("(recorded clip -> mock STT transcription)");
    applyResponse(postVoice(blob));
  }

  const statusOk = response && response.status === "ok";

  return (
    <div className="app">
      <header>
        <h1>Voice-Enabled RAG</h1>
        <p className="subtitle">HH Goa 2026 · Shortlisting Task 2</p>
      </header>

      <section className="status-bar">
        {healthError && <span className="badge badge-down">backend unreachable</span>}
        {health && (
          <>
            <span className={`badge badge-${health.status}`}>{health.status}</span>
            <span className="muted">env: {health.app_env}</span>
            <span className="muted">
              vector_store: {health.components.vector_store} · embedder:{" "}
              {health.components.embedder}
            </span>
          </>
        )}
      </section>

      <section className="query-panel">
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question..."
          />
          <button type="submit" disabled={loading}>
            {loading ? "Asking..." : "Ask"}
          </button>
        </form>

        <div className="voice-row">
          <MicButton onComplete={handleRecording} disabled={loading} />
          <input
            type="text"
            className="mock-transcript"
            placeholder='Dev: drive /voice without a mic, e.g. "What is supervised learning?"'
            onKeyDown={(e) => {
              if (e.key === "Enter" && e.target.value.trim()) {
                setLastTranscript(`[mock STT override] ${e.target.value}`);
                applyResponse(postVoice(new Blob([new Uint8Array(1)]), e.target.value));
                e.target.value = "";
              }
            }}
          />
        </div>

        {error && <div className="error-box">{error}</div>}

        {response && (
          <div className={`answer-box ${statusOk ? "" : "refused"}`}>
            {lastTranscript && (
              <p className="transcript">
                <strong>You asked:</strong> {lastTranscript}
              </p>
            )}
            <h3>Answer {statusOk ? "" : `(${response.status})`}</h3>
            <p>{response.answer}</p>

            {response.sources?.length > 0 && (
              <div className="sources">
                <h4>Sources ({response.sources.length})</h4>
                <ol>
                  {response.sources.map((s) => (
                    <li key={s.chunk_id}>
                      <span className="source-title">{s.title}</span>{" "}
                      <span className="muted">
                        [{s.document_id} · {s.chunk_id} · score {s.score}]
                      </span>
                      <p className="source-text">{s.text}</p>
                    </li>
                  ))}
                </ol>
              </div>
            )}

            <div className="meta-row">
              <span
                className={`badge ${
                  response.grounded ? "badge-ok" : "badge-degraded"
                }`}
              >
                {response.grounded ? "grounded" : "not grounded"}
              </span>
              <span>confidence: {(response.confidence * 100).toFixed(0)}%</span>
              <span>latency: {response.latency_ms} ms</span>
              {response.error && <span className="muted">({response.error})</span>}
            </div>

            {response.timings && (
              <details className="timings">
                <summary>Pipeline stage timings</summary>
                <ul>
                  {Object.entries(response.timings)
                    .filter(([k]) => k !== "total_ms")
                    .map(([stage, ms]) => (
                      <li key={stage}>
                        {stage}: {ms} ms
                      </li>
                    ))}
                  <li>
                    <strong>total: {response.timings.total_ms} ms</strong>
                  </li>
                </ul>
              </details>
            )}
          </div>
        )}
      </section>

      <footer>
        <p className="muted">
          Pipeline: STT → query processing → guardrails → hybrid retrieval (dense +
          BM25, RRF) → reranker → context selection → LLM → grounding/citation
          checks → answer.
        </p>
      </footer>
    </div>
  );
}
