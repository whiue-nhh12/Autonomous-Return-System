"""User-scoped continuation state for clarification turns."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Iterator, Protocol
from uuid import UUID

from pydantic import Field

from agent.core.state.shared import ReturnIntent, StateModel


class ConversationSnapshot(StateModel):
    """Minimal user-message context retained only while clarification is pending."""

    messages: list[str] = Field(default_factory=list, max_length=8)
    pending_intent: ReturnIntent | None = None
    claim: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationStore(Protocol):
    def turn_lock(
        self, user_id: UUID, conversation_id: str
    ) -> AbstractContextManager[None]: ...
    def get(self, user_id: UUID, conversation_id: str) -> ConversationSnapshot | None: ...
    def save(self, user_id: UUID, conversation_id: str, snapshot: ConversationSnapshot) -> None: ...
    def delete(self, user_id: UUID, conversation_id: str) -> None: ...


class InMemoryConversationStore:
    """Single-process continuation store with TTL and serialized turns per thread.

    Hosts running multiple processes should inject a shared store implementing
    ``ConversationStore`` and a distributed ``turn_lock``.
    """

    def __init__(self, *, ttl_seconds: int = 1800) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.ttl = timedelta(seconds=ttl_seconds)
        self._guard = RLock()
        self._turn_locks: dict[tuple[UUID, str], RLock] = {}
        self._turn_lock_users: dict[tuple[UUID, str], int] = {}
        self._snapshots: dict[tuple[UUID, str], ConversationSnapshot] = {}

    @contextmanager
    def turn_lock(self, user_id: UUID, conversation_id: str) -> Iterator[None]:
        key = (user_id, conversation_id)
        with self._guard:
            lock = self._turn_locks.get(key)
            if lock is None:
                lock = RLock()
                self._turn_locks[key] = lock
            self._turn_lock_users[key] = self._turn_lock_users.get(key, 0) + 1
        try:
            with lock:
                yield
        finally:
            with self._guard:
                users = self._turn_lock_users.get(key, 0) - 1
                if users <= 0:
                    self._turn_lock_users.pop(key, None)
                    self._turn_locks.pop(key, None)
                else:
                    self._turn_lock_users[key] = users

    def get(self, user_id: UUID, conversation_id: str) -> ConversationSnapshot | None:
        key = (user_id, conversation_id)
        with self._guard:
            now = datetime.now(timezone.utc)
            self._prune_expired(now)
            snapshot = self._snapshots.get(key)
            if snapshot is None:
                return None
            return snapshot.model_copy(deep=True)

    def save(self, user_id: UUID, conversation_id: str, snapshot: ConversationSnapshot) -> None:
        key = (user_id, conversation_id)
        refreshed = snapshot.model_copy(
            update={"updated_at": datetime.now(timezone.utc)}, deep=True
        )
        with self._guard:
            self._prune_expired(refreshed.updated_at)
            self._snapshots[key] = refreshed

    def delete(self, user_id: UUID, conversation_id: str) -> None:
        with self._guard:
            self._snapshots.pop((user_id, conversation_id), None)

    def _prune_expired(self, now: datetime) -> None:
        expired = [
            key
            for key, snapshot in self._snapshots.items()
            if now - snapshot.updated_at > self.ttl
        ]
        for key in expired:
            self._snapshots.pop(key, None)
