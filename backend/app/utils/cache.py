"""
Small in-memory TTL+LRU cache (Phase 4).

Used for query->response caching and query-embedding caching. Deliberately
dependency-free: an in-memory cache is preferred over Redis until there is a
reason to share cache state across processes.

Thread-safety: guarded with a lock since retrievers run in worker threads.
"""
from __future__ import annotations

import threading
import time
from typing import Any


class TTLCache:
    def __init__(self, maxsize: int = 512, ttl_seconds: float = 300.0) -> None:
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        # Counters are exposed via /api/v1/metrics.
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, value = entry
            if now - stored_at > self.ttl_seconds:
                del self._data[key]
                self.misses += 1
                return None
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._data) >= self.maxsize and key not in self._data:
                # Evict oldest entries first.
                for old_key in sorted(self._data, key=lambda k: self._data[k][0])[
                    : max(1, len(self._data) - self.maxsize + 1)
                ]:
                    del self._data[old_key]
            self._data[key] = (time.monotonic(), value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._data),
                "hits": self.hits,
                "misses": self.misses,
            }
