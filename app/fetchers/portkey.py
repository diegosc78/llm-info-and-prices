from __future__ import annotations

import asyncio
import logging

from ..models import SourcePayload
from .base import BaseFetcher

logger = logging.getLogger(__name__)

PRICING_REPO = "https://raw.githubusercontent.com/Portkey-AI/models/main"
# Provider names where we ignore pricing (general-only providers don't map cleanly)
# We fetch the full provider list from these well-known ids, but tolerate failures.


class PortkeyFetcher(BaseFetcher):
    """Portkey provides pricing (cents/token) + general config from GitHub JSON files.

    Structure: pricing/{provider}.json and general/{provider}.json, each a flat
    map of model-name -> config. Price values are per *single token* already
    (their docs: costDollars = tokens * price / 100, price is in cents/token).
    """

    name = "portkey"

    _PROVIDERS = [
        "openai", "anthropic", "google", "azure-openai", "bedrock", "bedrock-mantle",
        "mistral-ai", "cohere", "together-ai", "groq", "deepseek", "fireworks-ai",
        "perplexity-ai", "anyscale", "deepinfra", "cerebras", "x-ai", "reka-ai",
        "sagemaker", "vertex-ai", "palm", "stability-ai", "novita-ai", "moonshot",
        "minimax", "zhipu", "jina", "nomic", "voyage", "deepgram", "azure-ai",
        "workers-ai", "ai21", "oracle", "scenario-ai", "segmind", "monsterapi",
        "nebius", "lightning-ai", "dashscope", "predibase", "ember-cloud",
        "claude-platform-aws", "fireworks", "fal-ai", "grok",
    ]

    async def _fetch(self) -> SourcePayload:
        items: dict[str, dict] = {}
        raw = {"pricing": {}, "general": {}}
        sem = asyncio.Semaphore(8)

        async def grab(kind: str, provider: str) -> tuple[str, dict] | None:
            url = f"{PRICING_REPO}/{kind}/{provider}.json"
            async with sem:
                try:
                    resp = await self._request("GET", url)
                    data = resp.json()
                    if not isinstance(data, dict):
                        return None
                    return provider, data
                except Exception as exc:  # noqa: BLE001
                    logger.debug("portkey %s/%s failed: %s", kind, provider, exc)
                    return None

        tasks = []
        for provider in self._PROVIDERS:
            for kind in ("pricing", "general"):
                tasks.append(grab(kind, provider))
        results = await asyncio.gather(*tasks)

        # Reassemble per kind
        results_by_kind: dict[str, dict[str, dict]] = {
            "pricing": {}, "general": {}
        }
        # tasks were created as [pricing, general] pairs per provider; results
        # align with the creation order.
        for i, provider in enumerate(self._PROVIDERS):
            pricing_res = results[i * 2]
            general_res = results[i * 2 + 1]
            if pricing_res:
                results_by_kind["pricing"][pricing_res[0]] = pricing_res[1]
            if general_res:
                results_by_kind["general"][general_res[0]] = general_res[1]
        raw = results_by_kind

        if not raw["pricing"] and not raw["general"]:
            raise ValueError("Portkey: no provider files fetched")

        # Build per-model items from merged pricing+general
        providers = set(raw["pricing"]) | set(raw["general"])
        for provider in providers:
            pricing = raw["pricing"].get(provider, {})
            general = raw["general"].get(provider, {})
            model_keys = set(pricing) | set(general)
            for model in model_keys:
                if model == "default":
                    continue
                entry: dict = {
                    "portkey_provider": provider,
                    "portkey_model": model,
                }
                if model in pricing:
                    entry["portkey_pricing"] = pricing[model]
                if model in general:
                    entry["portkey_general"] = general[model]
                items[f"portkey/{provider}/{model}"] = entry

        return SourcePayload(name=self.name, raw=raw, items=items)
