export default function Footer() {
  return (
    <footer className="honor-strip">
      <div className="honor-inner">
        <div className="honor-badge">
          <span className="spark" aria-hidden="true">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2l2.4 6.9L21 11l-6.6 2.1L12 20l-2.4-6.9L3 11l6.6-2.1L12 2z" />
            </svg>
          </span>
          In honor of&nbsp;
          <span className="honor-org">Hacker House Goa 2026</span>
          &nbsp;— organized by&nbsp;
          <span className="honor-org">2:47 PM Studio</span>
        </div>
        <div className="honor-host">
          Hosted on <b>Devfolio</b>
        </div>

        <div className="team-block">
          <div className="team-name">
            <span className="team-kicker">Team</span> HS&#8209;TECH
          </div>
          <div className="team-members">
            <span className="member-chip">
              <strong>Harisudhan B</strong>
              <em>Backend Developer</em>
            </span>
            <span className="member-chip">
              <strong>Avinash R</strong>
              <em>DevOps</em>
            </span>
            <span className="member-chip">
              <strong>Aravindan P</strong>
              <em>Frontend Developer</em>
            </span>
          </div>
          <div className="team-institute">
            Manakula Vinayagar Institute of Technology
          </div>
          <p className="team-thanks">
            With heartfelt gratitude to <b>2:47 PM Studio</b>, <b>Devfolio</b> and every
            volunteer of <b>Hacker House Goa 2026</b> — thank you for building the space,
            energy and community that made this project possible.
          </p>
        </div>
      </div>

      <div className="site-footer container">
        <p>
          Voice → STT → query processing → guardrails → hybrid retrieval (dense + BM25,
          RRF fusion) → cross-encoder reranking → context selection → LLM generation →
          grounding &amp; citation validation → structured answer.
        </p>
      </div>
    </footer>
  );
}
