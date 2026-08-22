const STATUS_NOTES = {
  refused_off_topic: "This question falls outside the currently indexed knowledge base, so no answer was generated.",
  refused_unsafe: "Blocked by the input safety guardrail.",
  refused_no_context: "Retrieval did not surface sufficiently relevant context to answer reliably.",
  grounding_failed: "The generated answer failed grounding validation and was withheld.",
};

function ConfidenceGauge({ value }) {
  const pct = Math.max(0, Math.min(1, Number(value) || 0));
  const r = 20;
  const c = 2 * Math.PI * r;
  return (
    <div className="gauge" title="Answer confidence">
      <svg className="gauge-ring" width="48" height="48" viewBox="0 0 48 48">
        <defs>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#22d3ee" />
            <stop offset="100%" stopColor="#818cf8" />
          </linearGradient>
        </defs>
        <circle className="gauge-track" cx="24" cy="24" r={r} fill="none" strokeWidth="4.5" />
        <circle
          className="gauge-fill"
          cx="24"
          cy="24"
          r={r}
          fill="none"
          strokeWidth="4.5"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - pct)}
        />
      </svg>
      <span className="gauge-num">
        <strong>{Math.round(pct * 100)}%</strong> confidence
      </span>
    </div>
  );
}

export default function AnswerCard({ response }) {
  const ok = response.status === "ok";
  const refused = !ok && response.answer;
  const note = STATUS_NOTES[response.status];

  return (
    <article className={`answer-card ${ok ? "" : "refused"}`}>
      <div className="answer-head">
        <h3>{ok ? "Grounded answer" : "No answer generated"}</h3>
        <span className={`status-chip ${ok ? "ok" : "warn"}`}>{response.status}</span>
      </div>

      {note && <p className="answer-status-note">{note}</p>}

      <div className="answer-body">{refused || response.answer}</div>

      <div className="answer-foot">
        <ConfidenceGauge value={response.confidence} />
        <div className="badges">
          <span
            className={`status-chip ${response.grounded ? "ok" : "warn"}`}
            title="Every cited chunk was validated against retrieval results"
          >
            {response.grounded ? "grounded ✓" : "not grounded"}
          </span>
          {response.citations?.length > 0 && (
            <span className="status-chip ok" style={{ background: "var(--accent-soft)", borderColor: "rgba(34,211,238,.3)", color: "#a9e9f7" }}>
              {response.citations.length} citation{response.citations.length > 1 ? "s" : ""}
            </span>
          )}
        </div>
        <span className="latency-chip">{Math.round(response.latency_ms)} ms</span>
      </div>
    </article>
  );
}
