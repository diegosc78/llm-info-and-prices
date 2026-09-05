from __future__ import annotations

import asyncio
import time
from typing import Any


class TTLCache:
    """Simple thread-safe in-memory TTL cache (no Redis needed for a single-process service)."""

    def __init__(self, default_ttl: float = 3600):
        self._default_ttl = default_ttl
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        ttl = self._default_ttl if ttl is None else ttl
        async with self._lock:
            self._store[key] = (time.monotonic() + ttl, value)

    async def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            async with self._lock:
                self._store.pop(key, None)
            return None
        return value

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
