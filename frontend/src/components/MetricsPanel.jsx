const LABELS = {
  stt: "Speech-to-text",
  query_processing: "Query processing",
  input_guardrails: "Input guardrails",
  retrieval: "Hybrid retrieval",
  rerank: "Reranking",
  context_selection: "Context selection",
  generation: "Generation",
  output_guardrails: "Output guardrails",
};

export default function MetricsPanel({ response }) {
  const timings = response?.timings || {};
  const entries = Object.entries(timings)
    .filter(([k]) => k !== "total_ms")
    .sort((a, b) => b[1] - a[1]);

  if (!entries.length) {
    return (
      <aside className="rail-panel">
        <div className="panel-heading">
          <h3>Latency breakdown</h3>
        </div>
        <p className="rail-empty">Stage timings appear here after a query.</p>
      </aside>
    );
  }

  const max = Math.max(...entries.map(([, ms]) => ms), 1);
  const total =
    timings.total_ms ?? entries.reduce((acc, [, ms]) => acc + ms, 0);

  return (
    <aside className="rail-panel">
      <div className="panel-heading">
        <h3>Latency breakdown</h3>
        <span className="panel-count">ms</span>
      </div>
      {entries.map(([key, ms]) => {
        const base = key.replace(/_ms$/, "");
        return (
          <div className="metric-row" key={key}>
            <div className="metric-label">
              <span>{LABELS[base] || base}</span>
              <span className="ms">{ms >= 100 ? Math.round(ms) : ms.toFixed(1)}</span>
            </div>
            <div className="metric-bar">
              <i style={{ width: `${Math.max(2, (ms / max) * 100)}%` }} />
            </div>
          </div>
        );
      })}
      <div className="metric-total">
        <span>Total pipeline</span>
        <span className="ms">{Math.round(total)} ms</span>
      </div>
    </aside>
  );
}
