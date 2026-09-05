from __future__ import annotations

import logging

from ..models import SourcePayload
from .base import BaseFetcher

logger = logging.getLogger(__name__)

CLOUDPRICE_BASE = "https://ai.cloudprice.net/api/v1"


class CloudPriceFetcher(BaseFetcher):
    """CloudPrice provides a free public AI model pricing API.

    Primary: GET /litellm_model_prices.json (flat map, per-token costs)
    Enrichment: GET /models (paginated) for ids/aliases, descriptions, context windows.
    """

    name = "cloudprice"

    async def _fetch(self) -> SourcePayload:
        items: dict[str, dict] = {}
        raw: dict = {}

        # 1) Flat LiteLLM-style map (per-token costs already normalized)
        flat_resp = await self._request(
            "GET", f"{CLOUDPRICE_BASE}/litellm_model_prices.json"
        )
        flat = flat_resp.json()
        if not isinstance(flat, dict):
            raise ValueError("CloudPrice: expected object for litellm_model_prices.json")
        flat_models = {k: v for k, v in flat.items() if isinstance(v, dict)}
        raw["litellm_model_prices"] = flat
        for key, value in flat_models.items():
            entry = dict(value)
            entry["cloudprice_key"] = key
            entry["cloudprice_source"] = "litellm_model_prices"
            items[key] = entry

        # 2) Enrich with models catalog (aliases, descriptions, context windows).
        #    Aliases are kept as metadata on the canonical entry, NOT as separate
        #    items — otherwise each provider-variant would fragment into its own
        #    cluster. The merge engine uses these aliases to link other sources.
        next_token = None
        fetched = 0
        while True:
            params = {"include": "pricing", "page_size": 200}
            if next_token:
                params["next_token"] = next_token
            resp = await self._request("GET", f"{CLOUDPRICE_BASE}/models", params=params)
            body = resp.json()
            catalog = body.get("data", []) if isinstance(body, dict) else []
            raw.setdefault("models_catalog", []).append(catalog)
            for m in catalog:
                m = dict(m)
                mid = m.get("id")
                if not mid:
                    continue
                entry = dict(m)
                entry["cloudprice_source"] = "models_catalog"
                entry["cloudprice_aliases"] = m.get("ids") or [mid]
                items.setdefault(mid, {}).update(entry)
            fetched += len(catalog)
            meta = body.get("meta", {}) if isinstance(body, dict) else {}
            next_token = meta.get("next_token") if isinstance(meta, dict) else None
            if not next_token or fetched >= 15000:
                break

        if not items:
            raise ValueError("CloudPrice: no models fetched")
        for k in list(items):
            items[k]["cloudprice_total_hits"] = fetched
        return SourcePayload(name=self.name, raw=raw, items=items)