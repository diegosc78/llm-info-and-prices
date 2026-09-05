from __future__ import annotations

import logging

from ..models import SourcePayload
from .base import BaseFetcher

logger = logging.getLogger(__name__)


class OpenRouterFetcher(BaseFetcher):
    """OpenRouter models catalog: list all models and their properties."""

    name = "openrouter"

    async def _fetch(self) -> SourcePayload:
        headers: dict[str, str] = {}
        if self.settings.openrouter_api_key:
            headers["Authorization"] = f"Bearer {self.settings.openrouter_api_key}"

        items: dict[str, dict] = {}
        payload: dict = {"data": [], "total_count": 0, "links": {}}

        # OpenRouter: fetch full list (no pagination needed unless total>500)
        offset = 0
        limit = 1000
        while True:
            resp = await self._request(
                "GET",
                f"{self.settings.openrouter_api_url}/models",
                params={"offset": offset, "limit": limit},
                headers=headers,
            )
            body = resp.json()
            data = body.get("data", [])
            for model in data:
                m = dict(model)
                mid = m.get("id") or m.get("canonical_slug")
                if not mid:
                    continue
                items[mid] = m
            payload["data"].extend(data)
            payload["total_count"] = body.get("total_count", 0)
            payload["links"] = body.get("links", {})
            if not data:
                break
            total = body.get("total_count", 0)
            offset += len(data)
            if offset >= total or offset >= 5000:
                break

        if not items:
            raise ValueError("OpenRouter: no models fetched")
        return SourcePayload(name=self.name, raw=payload, items=items)