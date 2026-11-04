"""Small thread-safe TTL cache used by query augmentation features."""

from __future__ import annotations

from threading import Lock
from time import monotonic
from typing import Any


class TTLCache:
    def __init__(self, *, ttl_seconds: float, max_entries: int) -> None:
        if ttl_seconds <= 0 or max_entries <= 0:
            raise ValueError("cache settings must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._values: dict[object, tuple[float, Any]] = {}
        self._lock = Lock()

    def get(self, key: object) -> tuple[bool, Any]:
        with self._lock:
            entry = self._values.get(key)
            if entry is None:
                return False, None
            expires_at, value = entry
            if expires_at <= monotonic():
                del self._values[key]
                return False, None
            return True, value

    def put(self, key: object, value: Any) -> None:
        with self._lock:
            now = monotonic()
            self._values = {
                existing_key: entry
                for existing_key, entry in self._values.items()
                if entry[0] > now
            }
            if len(self._values) >= self.max_entries:
                oldest_key = min(self._values, key=lambda item: self._values[item][0])
                del self._values[oldest_key]
            self._values[key] = (now + self.ttl_seconds, value)