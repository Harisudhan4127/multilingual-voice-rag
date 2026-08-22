import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Microphone capture button (Phase 8).
 *
 * Uses MediaRecorder; clicking starts/stops recording and hands the blob to
 * onComplete. Recording state is surfaced so App can disable other controls.
 */
export default function MicButton({ onComplete, disabled = false }) {
  const [state, setState] = useState("idle"); // idle | requesting | recording | error
  const [error, setError] = useState(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);

  useEffect(() => {
    return () => {
      // Unmount cleanup: stop any in-flight recording.
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        recorderRef.current.stop();
      }
      recorderRef.current?.stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const startRecording = useCallback(async () => {
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
        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        setState("idle");
        onComplete(blob);
      };
      recorderRef.current = recorder;
      recorder.start();
      setState("recording");
    } catch (e) {
      setError(
        e.name === "NotAllowedError"
          ? "Microphone permission denied."
          : `Microphone unavailable: ${e.message}`
      );
      setState("error");
    }
  }, [onComplete]);

  const stopRecording = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
  }, []);

  const handleClick = () => {
    if (state === "recording") stopRecording();
    else if (state !== "requesting") startRecording();
  };

  const label =
    state === "recording"
      ? "Stop"
      : state === "requesting"
        ? "..."
        : "Record";

  return (
    <div className="mic-wrapper">
      <button
        type="button"
        className={`mic-button ${state === "recording" ? "recording" : ""}`}
        onClick={handleClick}
        disabled={disabled || state === "requesting"}
        title="Record a question"
      >
        {state === "recording" && <span className="pulse" aria-hidden="true" />}
        <span className="mic-icon" aria-hidden="true">
          {state === "recording" ? "\u23F8" : "\uD83C\uDFA4"}
        </span>
        <span>{label}</span>
      </button>
      {error && <span className="mic-error">{error}</span>}
    </div>
  );
}
