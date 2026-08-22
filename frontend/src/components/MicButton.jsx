import { useEffect, useRef, useState } from "react";

export default function MicButton({ onComplete, disabled = false }) {
  const [state, setState] = useState("idle");
  const [error, setError] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  useEffect(() => {
    return () => {
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        recorderRef.current.stop();
      }
      recorderRef.current?.stream?.getTracks().forEach((t) => t.stop());
      clearInterval(timerRef.current);
    };
  }, []);

  const startTimer = () => {
    setElapsed(0);
    timerRef.current = setInterval(
      () => setElapsed((s) => s + 1),
      1000
    );
  };

  const stopTimer = () => {
    clearInterval(timerRef.current);
    timerRef.current = null;
    setElapsed(0);
  };

  async function startRecording() {
    setError(null);
    setState("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        stopTimer();
        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        setState("idle");
        onComplete(blob);
      };
      recorderRef.current = recorder;
      recorder.start();
      startTimer();
      setState("recording");
    } catch (e) {
      setError(
        e.name === "NotAllowedError"
          ? "Microphone permission denied"
          : `Mic unavailable: ${e.message}`
      );
      setState("error");
    }
  }

  function handleClick() {
    if (state === "recording") {
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        recorderRef.current.stop();
      }
    } else if (state !== "requesting") {
      startRecording();
    }
  }

  const mmss = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;

  return (
    <div className="mic-wrap" style={{ display: "flex", alignItems: "center", gap: "0.55rem" }}>
      <button
        type="button"
        className={`mic-fab ${state === "recording" ? "recording" : ""}`}
        onClick={handleClick}
        disabled={disabled || state === "requesting"}
        title={state === "recording" ? "Stop recording" : "Record a question"}
        aria-label={state === "recording" ? "Stop recording" : "Start voice recording"}
      >
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          {state === "recording" ? (
            <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none" />
          ) : (
            <>
              <rect x="9" y="2" width="6" height="12" rx="3" />
              <path d="M5 10v1a7 7 0 0 0 14 0v-1" />
              <line x1="12" y1="18" x2="12" y2="22" />
            </>
          )}
        </svg>
      </button>
      {state === "recording" && <span className="mic-timer">{mmss}</span>}
      {error && (
        <span style={{ fontSize: "0.74rem", color: "var(--danger)" }}>{error}</span>
      )}
    </div>
  );
}
