export default function SourcesPanel({ sources }) {
  return (
    <aside className="rail-panel">
      <div className="panel-heading">
        <h3>Sources</h3>
        {sources?.length > 0 && <span className="panel-count">{sources.length}</span>}
      </div>

      {!sources?.length && (
        <p className="rail-empty">Cited passages appear here after a query.</p>
      )}

      {sources?.map((s, i) => {
        const score = Math.min(1, Math.max(0, s.score));
        return (
          <div className="source-card" key={s.chunk_id}>
            <div className="source-top">
              <span className="source-idx">{i + 1}</span>
              <span className="source-name" title={s.title}>{s.title}</span>
              <span className="score-val">{score.toFixed(2)}</span>
            </div>
            <div className="score-bar">
              <i style={{ width: `${Math.max(4, score * 100)}%` }} />
            </div>
            <p className="source-text">{s.text}</p>
            <span className="source-chunkid">{s.chunk_id}</span>
          </div>
        );
      })}
    </aside>
  );
}
