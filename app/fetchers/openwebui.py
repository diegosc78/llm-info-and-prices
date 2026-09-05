from __future__ import annotations

import asyncio
import logging

from ..models import SourcePayload
from .base import BaseFetcher

logger = logging.getLogger(__name__)


class OpenWebUIFetcher(BaseFetcher):
    """Fetches the user's OpenWebUI model list.

    Tries several known endpoints (tolerant):
      1. {base}/api/models/list  (the pattern the user provided)
      2. {base}/api/models
      3. {base}/api/v1/models/
    Then enriches each model via {base}/api/models/model?id={id}.
    """

    name = "openwebui"

    LIST_ENDPOINTS = ("/api/models/list", "/api/models", "/api/v1/models/")

    async def _fetch(self) -> SourcePayload:
        base = (self.settings.openwebui_api_url or "").rstrip("/")
        if not base:
            raise ValueError("OPENWEBUI_API_URL not configured")

        headers: dict[str, str] = {}
        if self.settings.openwebui_api_key:
            headers["Authorization"] = f"Bearer {self.settings.openwebui_api_key}"

        raw: dict = {}
        models: list[dict] = []
        errors: list[str] = []

        for endpoint in self.LIST_ENDPOINTS:
            try:
                resp = await self._request("GET", f"{base}{endpoint}", headers=headers)
                body = resp.json()
                raw[endpoint] = body
                models = self._extract_models(body)
                if models:
                    logger.info("openwebui: using %s (%d models)", endpoint, len(models))
                    break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{endpoint}: {exc}")
                continue

        if not models:
            raise ValueError("OpenWebUI: no model source worked: " + "; ".join(errors))

        items: dict[str, dict] = {}
        for m in models:
            mid = m.get("id")
            if not mid:
                continue
            items[mid] = {"openwebui_source": "list", **m}

        # Enrich with detail endpoint when available (best-effort, parallelized
        # but rate-limited so we don't hammer the OpenWebUI instance).
        sem = asyncio.Semaphore(12)

        async def limited(m):
            async with sem:
                return await self._fetch_detail(m, base, headers)

        detail_tasks = [
            limited(m) for m in models if m.get("id")
        ]
        if detail_tasks:
            await asyncio.gather(*detail_tasks, return_exceptions=True)
        raw.setdefault("detail_endpoint_error", None)

        return SourcePayload(name=self.name, raw=raw, items=items)

    async def _fetch_detail(
        self, model: dict, base: str, headers: dict
    ) -> str | None:
        mid = model.get("id")
        if not mid:
            return None
        for endpoint in (f"/api/models/model?id={mid}", f"/api/models/{mid}"):
            try:
                resp = await self._request(
                    "GET", f"{base}{endpoint}", headers=headers
                )
                detail = resp.json()
                if isinstance(detail, dict):
                    model["openwebui_detail"] = detail
                return endpoint
            except Exception:  # noqa: BLE001
                continue
        return None

    def _extract_models(self, body) -> list[dict]:
        if isinstance(body, dict):
            # common shapes: {data: [...]} or {models: [...]} or {data:{models:[...]}}
            for key in ("data", "models", "model_list", "list"):
                val = body.get(key)
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    return val
            return []
        if isinstance(body, list) and body and isinstance(body[0], dict):
            return body
        return []