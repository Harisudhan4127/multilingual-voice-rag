"""
Lightweight process-local metrics collector (Section 22 / Phase 7).

Counters + latency aggregates for /api/v1/metrics. Attached to app.state at
startup -- no global mutable state. Good enough for a single-process demo;
a production deployment would export these to Prometheus instead.
"""
from __future__ import annotations

import time
from collections import Counter, deque


class Metrics:
    MAX_LATENCY_SAMPLES = 500

    def __init__(self) -> None:
        self.started_at = time.time()
        self.requests_total = 0
        self.voice_requests_total = 0
        self.status_counts: Counter[str] = Counter()
        self.errors_total = 0
        self.refusals_total = 0
        self.cache_hits_total = 0
        self._latencies_ms: deque[float] = deque(maxlen=self.MAX_LATENCY_SAMPLES)
        self._stage_totals_ms: dict[str, float] = {}
        self._stage_counts: Counter[str] = Counter()

    def record_request(self, voice: bool = False) -> None:
        self.requests_total += 1
        if voice:
            self.voice_requests_total += 1

    def record_response(
        self,
        status: str,
        latency_ms: float,
        timings: dict[str, float],
        *,
        refused: bool = False,
        errored: bool = False,
        cache_hit: bool = False,
    ) -> None:
        self.status_counts[status] += 1
        self._latencies_ms.append(latency_ms)
        if refused:
            self.refusals_total += 1
        if errored:
            self.errors_total += 1
        if cache_hit:
            self.cache_hits_total += 1
        for stage, ms in timings.items():
            base = stage.replace("_ms", "")
            self._stage_totals_ms[f"{base}_ms"] = (
                self._stage_totals_ms.get(f"{base}_ms", 0.0) + ms
            )
            self._stage_counts[base] += 1

    @staticmethod
    def _percentile(values: list[float], pct: float) -> float:
        from app.utils.timing import percentile

        return percentile(values, pct)

    def snapshot(self) -> dict:
        latencies = list(self._latencies_ms)
        avg = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        stage_avgs = {
            key: round(total / self._stage_counts[key.replace("_ms", "")], 2)
            for key, total in self._stage_totals_ms.items()
            if self._stage_counts.get(key.replace("_ms", ""))
        }
        return {
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "requests_total": self.requests_total,
            "voice_requests_total": self.voice_requests_total,
            "responses_by_status": dict(self.status_counts),
            "refusals_total": self.refusals_total,
            "errors_total": self.errors_total,
            "cache_hits_total": self.cache_hits_total,
            "latency_ms": {
                "samples": len(latencies),
                "avg": avg,
                "p50": self._percentile(latencies, 50),
                "p70": self._percentile(latencies, 70),
                "p95": self._percentile(latencies, 95),
                "p100": max(latencies) if latencies else 0.0,
            },
            "avg_stage_latency_ms": stage_avgs,
        }
