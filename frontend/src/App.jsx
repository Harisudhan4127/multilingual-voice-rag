import { useEffect, useState } from "react";
import Header from "./components/Header";
import PipelineFlow from "./components/PipelineFlow";
import QueryComposer from "./components/QueryComposer";
import AnswerCard from "./components/AnswerCard";
import SourcesPanel from "./components/SourcesPanel";
import MetricsPanel from "./components/MetricsPanel";
import Footer from "./components/Footer";
import { getHealth, postQuery, postVoice } from "./services/api";
import "./App.css";

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);

  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [askedViaVoice, setAskedViaVoice] = useState(false);
  const [transcript, setTranscript] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((e) => setHealthError(e.message));
  }, []);

  async function run(promise) {
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

  function submitText(q) {
    if (!q.trim()) return;
    setAskedViaVoice(false);
    setTranscript(null);
    run(postQuery(q));
  }

  function submitVoice(blob) {
    setAskedViaVoice(true);
    setTranscript({ text: "Recorded clip → STT transcription", pending: true });
    run(postVoice(blob));
  }

  function submitMockVoice(text) {
    setAskedViaVoice(true);
    setTranscript({ text: `[mock STT] ${text}`, pending: false });
    run(postVoice(new Blob([new Uint8Array(1)]), text));
  }

  return (
    <div className="app-shell">
      <div className="backdrop" aria-hidden="true">
        <div className="grid-overlay" />
      </div>

      <Header health={health} healthError={healthError} />

      <section className="hero container">
        <span className="hero-kicker">Retrieval-Augmented Intelligence</span>
        <h1 className="hero-title">
          Ask by voice. Get answers that are{" "}
          <span className="grad-text">grounded &amp; cited.</span>
        </h1>
        <p className="hero-sub">
          VaaniRAG fuses dense vector search with BM25, reranks with a cross-encoder,
          and refuses to answer when the evidence isn&apos;t there.
        </p>
      </section>

      <PipelineFlow
        response={response}
        loading={loading}
        isVoice={askedViaVoice}
      />

      <main className="container workspace">
        <div className="col-main">
          <QueryComposer
            query={query}
            onQueryChange={setQuery}
            onSubmit={submitText}
            onVoice={submitVoice}
            onMockVoice={submitMockVoice}
            loading={loading}
          />

          {transcript && (
            <span className="transcript-pill">
              🎙 <strong>{transcript.text}</strong>
            </span>
          )}

          {error && (
            <div className="error-banner">
              <span>⚠</span>
              <span>{error}</span>
            </div>
          )}

          {loading && (
            <div className="thinking-card">
              <div className="ticker">
                <span className="ticker-dot" />
                Running the RAG pipeline…
              </div>
              <div className="skel w60" />
              <div className="skel w90" />
              <div className="skel w75" />
              <div className="skel w90" />
            </div>
          )}

          {!loading && response && <AnswerCard response={response} />}

          {!loading && !response && !error && (
            <div className="empty-state">
              <div className="empty-orb" aria-hidden="true">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
                  <rect x="4" y="9" width="2.6" height="6" rx="1.3" fill="currentColor" />
                  <rect x="8.5" y="5" width="2.6" height="14" rx="1.3" fill="currentColor" />
                  <rect x="13" y="7" width="2.6" height="10" rx="1.3" fill="currentColor" />
                  <rect x="17.5" y="10" width="2.6" height="4" rx="1.3" fill="currentColor" />
                </svg>
              </div>
              <h3 className="empty-title">Your knowledge base is listening</h3>
              <p className="empty-sub">
                Type a question, pick a suggestion, or hold the mic. Every answer cites
                the exact passages it was built from.
              </p>
              <div className="feature-grid">
                <div className="feature-card">
                  <h4>Voice-native</h4>
                  <p>Speech-to-text feeds the same grounded pipeline as typed queries.</p>
                </div>
                <div className="feature-card">
                  <h4>Hybrid retrieval</h4>
                  <p>Dense vectors + BM25 fused with RRF, then cross-encoder reranked.</p>
                </div>
                <div className="feature-card">
                  <h4>Honest by design</h4>
                  <p>Safety, off-topic and grounding guardrails refuse instead of hallucinate.</p>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="col-rail">
          <SourcesPanel sources={response?.sources || []} />
          <MetricsPanel response={response} />
        </div>
      </main>

      <Footer />
    </div>
  );
}
