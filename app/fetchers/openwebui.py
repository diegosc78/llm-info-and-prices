from __future__ import annotations

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

    The list payload already contains the full model doc in ``info``,
    including ``info.base_model_id`` (the underlying base model for custom
    models/agents) and ``info.meta.description``.
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
            entry = {"openwebui_source": "list", **m}
            info = m.get("info") or {}
            base_model_id = info.get("base_model_id")
            # base models reference themselves or have no base; skip those.
            if isinstance(base_model_id, str) and base_model_id:
                if base_model_id.lower() != str(mid).lower():
                    entry["base_model_id"] = base_model_id
            # description/name travel inside info
            meta = info.get("meta") or {}
            meta_desc = meta.get("description")
            if not entry.get("description") and isinstance(meta_desc, str):
                entry["description"] = meta_desc
            if not entry.get("name") and isinstance(info.get("name"), str):
                entry["name"] = info["name"]
            items[mid] = entry

        return SourcePayload(name=self.name, raw=raw, items=items)

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