"""
Timing helpers used by every pipeline stage (Phase 4).

Latency accounting convention across the codebase:
    start = time.perf_counter()
    ... work ...
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
"""
from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager


def now_ms() -> float:
    return time.perf_counter() * 1000


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


@contextmanager
def timed(timings: dict[str, float], key: str) -> Generator[None, None, None]:
    """Context manager that records stage latency into a timings dict."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = timings.get(key, 0.0) + round(
            (time.perf_counter() - start) * 1000, 2
        )


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile on a sorted-ascending copy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return round(ordered[idx], 2)


def latency_summary(values: list[float]) -> dict[str, float]:
    """min/avg/P50/P70/P95/P100 summary required by Section 24."""
    if not values:
        return {k: 0.0 for k in ("min", "avg", "p50", "p70", "p95", "p100")}
    return {
        "min": round(min(values), 2),
        "avg": round(sum(values) / len(values), 2),
        "p50": percentile(values, 50),
        "p70": percentile(values, 70),
        "p95": percentile(values, 95),
        "p100": round(max(values), 2),
    }
