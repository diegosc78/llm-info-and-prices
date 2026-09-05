from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import Settings, get_settings
from ..models import SourcePayload, SourceStatus

logger = logging.getLogger(__name__)


class HTTPError(Exception):
    pass


class BaseFetcher(ABC):
    """Base class for all data source fetchers.

    Each fetcher:
    - Fetches raw data from its source (with retries via tenacity).
    - Emits a SourcePayload with a dict of variant-id -> model dict.
    - Records its own SourceStatus so /health can report per-source health.
    """

    name: str = "base"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._status = SourceStatus(name=self.name)
        self._client: httpx.AsyncClient | None = None

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Perform an HTTP request with retries + exponential backoff.

        4xx responses are returned immediately (retrying `Model not found` is useless);
        network errors and 5xx are retried up to http_max_retries times.
        """
        attempts = self._retrier()
        last_exc: Exception | None = None
        async for attempt in attempts:
            with attempt:
                try:
                    resp = await self._get_client().request(method, url, **kwargs)
                    if resp.status_code >= 400 and resp.status_code < 500:
                        return resp  # no retry for client errors
                    resp.raise_for_status()
                    return resp
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    raise
        raise HTTPError(f"{self.name}: request failed: {last_exc}")

    def _retrier(self):
        return AsyncRetrying(
            stop=stop_after_attempt(self.settings.http_max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(
                (httpx.HTTPError, httpx.TimeoutException, HTTPError)
            ),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.settings.http_timeout_seconds,
                follow_redirects=True,
            )
        return self._client

    def _record_success(self, count: int) -> None:
        self._status.ok = True
        self._status.error = None
        self._status.models_count = count
        self._status.last_fetch = _now()
        self._status.stale = False
        self._last_success = time.monotonic()

    def _record_failure(self, exc: Exception, stale_ok: bool = False) -> None:
        self._status.ok = False
        self._status.error = str(exc)
        self._status.stale = getattr(self, "_last_success", None) is not None and stale_ok

    async def fetch(self) -> SourcePayload:
        """Fetch and transform raw source data. Never raises; degrades to stale/empty."""
        try:
            payload = await self._fetch()
            self._record_success(len(payload.items))
            return payload
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fetcher %s failed: %s", self.name, exc)
            self._record_failure(exc, stale_ok=True)
            return SourcePayload(name=self.name)

    @abstractmethod
    async def _fetch(self) -> SourcePayload:
        ...

    @property
    def status(self) -> SourceStatus:
        return self._status

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
