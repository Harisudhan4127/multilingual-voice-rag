export default function Header({ health, healthError }) {
  const up = Boolean(health) && !healthError;
  const label = up
    ? "Systems operational"
    : healthError
      ? "Backend unreachable"
      : "Connecting…";

  return (
    <header className="site-header">
      <div className="container header-inner">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <rect x="4" y="9" width="2.6" height="6" rx="1.3" fill="currentColor" />
              <rect x="8.5" y="5" width="2.6" height="14" rx="1.3" fill="currentColor" />
              <rect x="13" y="7" width="2.6" height="10" rx="1.3" fill="currentColor" />
              <rect x="17.5" y="10" width="2.6" height="4" rx="1.3" fill="currentColor" />
            </svg>
          </div>
          <div>
            <div className="brand-name">VaaniRAG</div>
            <div className="brand-tag">Voice Intelligence Platform</div>
          </div>
        </div>
        <div className={`status-pill ${up ? "up" : "down"}`}>
          <span className="status-dot" />
          {label}
        </div>
      </div>
    </header>
  );
}
