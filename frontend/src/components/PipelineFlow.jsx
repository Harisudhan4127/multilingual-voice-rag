import { useEffect, useState } from "react";

const STAGES = [
  { key: "stt", label: "Speech → Text" },
  { key: "query_processing", label: "Understand" },
  { key: "input_guardrails", label: "Safety In" },
  { key: "retrieval", label: "Retrieve" },
  { key: "rerank", label: "Rerank" },
  { key: "context_selection", label: "Context" },
  { key: "generation", label: "Generate" },
  { key: "output_guardrails", label: "Grounding" },
];

const TICKER_STEPS = [
  "Transcribing audio…",
  "Analyzing query…",
  "Fusing dense + BM25 results…",
  "Reranking passages…",
  "Selecting context…",
  "Generating answer…",
];

export default function PipelineFlow({ response, loading, isVoice }) {
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!loading) {
      setTick(0);
      return undefined;
    }
    const id = setInterval(() => setTick((t) => t + 1), 900);
    return () => clearInterval(id);
  }, [loading]);

  const timings = response?.timings || {};
  const total = timings.total_ms ?? response?.latency_ms ?? null;
  const stages = isVoice ? STAGES : STAGES.filter((s) => s.key !== "stt");

  function stageState(stage) {
    if (typeof timings[`${stage.key}_ms`] === "number") return "done";
    if (loading && stage.key === stages[tick % stages.length].key) return "active";
    return "pending";
  }

  return (
    <section className="container">
      <div className="pipeline-card">
        <div className="pipeline-head">
          <span className="pipeline-label">Live pipeline trace</span>
          {total != null && !loading && (
            <span className="pipeline-total">{Math.round(total)} ms total</span>
          )}
          {loading && (
            <span style={{ fontSize: "0.74rem", color: "var(--text-3)", fontFamily: "var(--font-mono)" }}>
              running…
            </span>
          )}
        </div>
        <div className="pipeline-track">
          {stages.map((stage) => {
            const state = stageState(stage);
            const ms = timings[`${stage.key}_ms`];
            return (
              <div className={`stage ${state}`} key={stage.key}>
                <div className="stage-dot" />
                <span className="stage-label">{stage.label}</span>
                <span className="stage-ms">{typeof ms === "number" ? `${ms.toFixed(1)} ms` : ""}</span>
              </div>
            );
          })}
        </div>
        {loading && (
          <p style={{ margin: "0.6rem 0 0.2rem", fontSize: "0.78rem", color: "var(--accent)" }}>
            {TICKER_STEPS[tick % TICKER_STEPS.length]}
          </p>
        )}
      </div>
    </section>
  );
}
