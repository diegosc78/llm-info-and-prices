from __future__ import annotations

from ..models import SourcePayload
from .base import BaseFetcher

UPSTREAM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/"
    "model_prices_and_context_window.json"
)


class LiteLLMUpstreamFetcher(BaseFetcher):
    name = "litellm_upstream"

    async def _fetch(self) -> SourcePayload:
        resp = await self._request("GET", UPSTREAM_URL)
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("LiteLLM upstream: expected a JSON object")

        items: dict[str, dict] = {}
        for key, value in data.items():
            if key in ("sample_spec", "_meta", "sample"):
                continue
            if not isinstance(value, dict):
                continue
            value = dict(value)
            value["litellm_key"] = key
            value["mode"] = value.get("mode", "chat")
            items[key] = value

        return SourcePayload(name=self.name, raw=data, items=items)
