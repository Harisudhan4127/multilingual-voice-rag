const BASE = "/api/v1";

async function handle(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ? JSON.stringify(body.detail) : detail;
    } catch {
      // response wasn't JSON; keep statusText
    }
    throw new Error(`Request failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${BASE}/health`);
  return handle(res);
}

export async function postQuery(query) {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  return handle(res);
}

// Voice path (Phase 8): audio blob -> /api/v1/voice multipart.
// `mockTranscript` is only honored by the backend when STT_PROVIDER=mock;
// for real providers it is ignored server-side.
export async function postVoice(audioBlob, mockTranscript = null) {
  const form = new FormData();
  const ext = audioBlob.type.includes("ogg") ? "ogg" : "webm";
  form.append("audio", audioBlob, `recording.${ext}`);
  if (mockTranscript && mockTranscript.trim()) {
    form.append("mock_transcript", mockTranscript.trim());
  }
  const res = await fetch(`${BASE}/voice`, {
    method: "POST",
    body: form,
  });
  return handle(res);
}
