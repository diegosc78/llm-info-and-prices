from __future__ import annotations

import asyncio
import logging
from typing import Any

from .cache import TTLCache
from .config import Settings, get_settings
from .fetchers import (
    CloudPriceFetcher,
    LiveBenchFetcher,
    LiteLLMUpstreamFetcher,
    LiteLLMUserFetcher,
    OpenRouterFetcher,
    OpenWebUIFetcher,
    PortkeyFetcher,
)
from .merge import MergeEngine, normalize_id
from .models import ModelData, SourcePayload, SourceStatus

logger = logging.getLogger(__name__)


class AppState:
    """Central in-memory store of fetched sources + merged models."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.cache = TTLCache(self.settings.cache_ttl_seconds)
        self.models: list[ModelData] = []
        self.sources: dict[str, SourceStatus] = {}
        self.payloads: dict[str, SourcePayload] = {}
        self._refresh_task: asyncio.Task | None = None
        self._latest_error: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    def _make_fetchers(self) -> list:
        return [
            LiteLLMUpstreamFetcher(self.settings),
            PortkeyFetcher(self.settings),
            CloudPriceFetcher(self.settings),
            OpenRouterFetcher(self.settings),
            LiveBenchFetcher(self.settings),
            LiteLLMUserFetcher(self.settings),
            OpenWebUIFetcher(self.settings),
        ]

    async def start(self) -> None:
        await self.refresh()
        self._refresh_task = asyncio.create_task(self._periodic_refresh())
        logger.info("AppState started with %d models", len(self.models))

    async def stop(self) -> None:
        self._closed = True
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        for fetcher in self._make_fetchers():
            await fetcher.aclose()

    async def refresh(self) -> dict[str, SourceStatus]:
        """Fetch all sources concurrently, then merge. Never raises."""
        fetchers = self._make_fetchers()
        payloads_list: list[SourcePayload] = []

        async def run(fetcher) -> SourcePayload:
            try:
                payload = await fetcher.fetch()
            except Exception as exc:  # noqa: BLE001
                logger.error("fetcher %s crashed: %s", fetcher.name, exc)
                return SourcePayload(name=fetcher.name)
            finally:
                await fetcher.aclose()
            return payload

        results = await asyncio.gather(*[run(f) for f in fetchers])

        async with self._lock:
            self.payloads = {}
            self.sources = {}
            for payload in results:
                self.payloads[payload.name] = payload
                # find the fetcher status (payloads come from same instance set)
            for f in fetchers:
                self.sources[f.name] = f.status
            merged = MergeEngine([p for p in results if p is not None]).build()
            self.models = merged
            self._resolve_base_prices()

        logger.info(
            "refresh complete: %d models, %d/%d sources ok",
            len(self.models),
            sum(1 for s in self.sources.values() if s.ok),
            len(self.sources),
        )
        return dict(self.sources)

    # context-window fields inherited from a base model when missing
    _WINDOW_FIELDS = (
        "context_length",
        "max_input_tokens",
        "max_output_tokens",
        "litellm_provider",
    )

    def _resolve_base_prices(self) -> None:
        """Resolve custom/proxy models to their underlying base model and
        inherit its pricing and context-window data.

        Chains like OpenWebUI 'traductor-tecnlogo' -> base 'migemini-2.5-flash'
        (LiteLLM proxy) -> origin 'openrouter/google/gemini-2.5-flash' are
        followed recursively until a model with concrete data is found. Pricing
        is filled from the deepest link that has any, and window metadata
        (max_input_tokens, max_output_tokens, context_length, litellm_provider)
        from the deepest link that carries it. Existing values are never
        overwritten, only inherited when missing.
        """
        if not self.models:
            return

        lookup: dict[str, ModelData] = {}
        for m in self.models:
            lookup[normalize_id(m.canonical_slug)] = m
            for a in m.aliases:
                lookup.setdefault(normalize_id(a), m)

        logger.info("resolving base-model price/context chains over %d models", len(self.models))

        for m in self.models:
            chain = self._base_chain(m, lookup)

            # deepest link of the chain with concrete pricing / window data
            price_target = next((c for c in reversed(chain) if c.pricing), None)
            window_target = next(
                (
                    c
                    for c in reversed(chain)
                    if any(getattr(c, f) for f in self._WINDOW_FIELDS)
                ),
                None,
            )
            if price_target is None and window_target is None:
                continue

            for c in chain:
                self._inherit_price(c, price_target)
                self._inherit_window(c, window_target)

    def _base_chain(
        self,
        m: ModelData,
        lookup: dict[str, ModelData],
    ) -> list[ModelData]:
        chain: list[ModelData] = []
        cur: ModelData | None = m
        seen: set[str] = set()
        while cur is not None:
            if cur.canonical_slug in seen:
                break  # cycle guard
            seen.add(cur.canonical_slug)
            chain.append(cur)
            base = cur.base_model_id
            if not base:
                break
            b = lookup.get(normalize_id(base))
            if b is None:
                stripped = self._strip_doubled_provider(base)
                if stripped and stripped != normalize_id(base):
                    b = lookup.get(stripped)
            if b is None:
                break
            cur = b
        return chain

    @staticmethod
    def _inherit_price(c: ModelData, target: ModelData | None) -> None:
        if target is None or c is target:
            return
        added = False
        for k, v in target.pricing.items():
            if k not in c.pricing:
                c.pricing[k] = v
                c.pricing_source[k] = f"derived from {target.canonical_slug}"
                added = True
        if added:
            c.resolved_price_from = target.canonical_slug

    @classmethod
    def _inherit_window(cls, c: ModelData, target: ModelData | None) -> None:
        if target is None or c is target:
            return
        derived = False
        for f in cls._WINDOW_FIELDS:
            cur = getattr(c, f, None)
            val = getattr(target, f, None)
            if cur is None and val:
                setattr(c, f, val)
                derived = True
        if derived:
            c.resolved_context_from = target.canonical_slug

    @staticmethod
    def _strip_doubled_provider(v: str) -> str:
        """openai/openai/gpt-oss-20b -> openai/gpt-oss-20b (noise in proxies)."""
        parts = normalize_id(v).split("/")
        if len(parts) >= 3 and parts[0] == parts[1]:
            return "/".join(parts[1:])
        return normalize_id(v)

    async def _periodic_refresh(self) -> None:
        interval = self.settings.refresh_interval_seconds
        logger.info("periodic refresh every %ss", interval)
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            if self._closed:
                break
            try:
                await self.refresh()
            except Exception as exc:  # noqa: BLE001
                logger.exception("periodic refresh failed: %s", exc)

    # ---- Query helpers ----
    def get_model(self, slug: str) -> ModelData | None:
        slug = slug.strip().lower()
        for m in self.models:
            if m.canonical_slug == slug:
                return m
            if slug in m.aliases:
                return m
        return None

    def search(
        self,
        q: str | None = None,
        provider: str | None = None,
        mode: str | None = None,
        user_instances: bool | None = None,
    ) -> list[ModelData]:
        q = (q or "").strip().lower()
        provider = (provider or "").strip().lower()
        mode = (mode or "").strip().lower()
        out = []
        for m in self.models:
            if user_instances is not None:
                in_user = bool(m.instances)
                if user_instances != in_user:
                    continue
            if provider and provider not in m.provider.lower():
                continue
            if mode and mode != m.mode.lower():
                continue
            if q:
                haystack = " ".join(
                    [
                        m.canonical_slug,
                        m.display_name,
                        m.description,
                        m.provider,
                        m.mode,
                        " ".join(m.aliases),
                        " ".join(m.urls.values()),
                    ]
                ).lower()
                if q not in haystack:
                    continue
            out.append(m)
        out.sort(key=lambda m: (m.canonical_slug))
        return out

    @property
    def total_models(self) -> int:
        return len(self.models)