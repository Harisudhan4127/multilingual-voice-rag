import { useState } from "react";
import MicButton from "./MicButton";

const SUGGESTIONS = [
  "What is supervised learning?",
  "How does photosynthesis work?",
  "Explain the significance of the Quit India Movement",
  "Best pizza toppings in the world?",
];

const SEARCH_ICON = (
  <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <circle cx="11" cy="11" r="7" />
    <line x1="21" y1="21" x2="16.5" y2="16.5" />
  </svg>
);

export default function QueryComposer({
  query,
  onQueryChange,
  onSubmit,
  onVoice,
  onMockVoice,
  loading,
}) {
  const [mockText, setMockText] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    onSubmit(query.trim());
  }

  return (
    <section className="composer">
      <form onSubmit={handleSubmit} style={{ margin: 0 }}>
        <div className="composer-row">
          <span className="composer-icon">{SEARCH_ICON}</span>
          <input
            className="composer-input"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="Ask anything — grounded answers come from your indexed knowledge base…"
            maxLength={2000}
            aria-label="Question"
            autoComplete="off"
          />
          <button className="send-btn" type="submit" disabled={loading || !query.trim()}>
            {loading ? "Thinking…" : "Ask"}
            {!loading && (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                <line x1="5" y1="12" x2="19" y2="12" />
                <polyline points="12 5 19 12 12 19" />
              </svg>
            )}
          </button>
        </div>
      </form>

      <div className="tools-row">
        <MicButton onComplete={onVoice} disabled={loading} />
        <input
          className="dev-input"
          value={mockText}
          onChange={(e) => setMockText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && mockText.trim()) {
              onMockVoice(mockText.trim());
              setMockText("");
            }
          }}
          placeholder='Dev: drive /voice without a mic — type a mock transcript and press Enter'
        />
      </div>

      <div className="chip-row" style={{ padding: "0 1.1rem 1rem" }}>
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            type="button"
            className="chip"
            onClick={() => onSubmit(s)}
            disabled={loading}
          >
            {s}
          </button>
        ))}
      </div>
    </section>
  );
}
