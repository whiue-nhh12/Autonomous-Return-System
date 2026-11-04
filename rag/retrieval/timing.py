"""Thread-safe timing accumulator for retrieval stages."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from threading import Lock
from time import perf_counter


class RetrievalTimings:
    """Aggregate named stage durations safely across retrieval worker threads."""

    def __init__(self) -> None:
        self._seconds: dict[str, float] = defaultdict(float)
        self._lock = Lock()

    def reset(self) -> None:
        with self._lock:
            self._seconds.clear()

    def record(self, name: str, elapsed_seconds: float) -> None:
        with self._lock:
            self._seconds[name] += elapsed_seconds

    @contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        started_at = perf_counter()
        try:
            yield
        finally:
            self.record(name, perf_counter() - started_at)

    def snapshot_ms(self) -> dict[str, float]:
        with self._lock:
            return {name: round(seconds * 1000, 2) for name, seconds in self._seconds.items()}
