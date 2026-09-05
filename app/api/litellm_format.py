from __future__ import annotations

from typing import Any

from ..models import ModelData

ACTION_KEYS = ("input_cost_per_token", "output_cost_per_token")
MAX_TOKENS_KEYS = ("max_input_tokens", "max_output_tokens")

# Known key allowlist for the LiteLLM cost map output format. Anything else is
# dropped to keep the payload clean and LiteLLM-compatible.
LITELLM_KEYS = {
    "input_cost_per_token",
    "output_cost_per_token",
    "input_cost_per_audio_token",
    "output_cost_per_audio_token",
    "input_cost_per_image",
    "output_cost_per_image",
    "input_cost_per_pixel",
    "output_cost_per_pixel",
    "input_cost_per_video_per_second",
    "input_cost_per_audio_per_second",
    "input_cost_per_query",
    "output_cost_per_query",
    "input_cost_per_1k_tokens_cache_hit",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
    "cache_read_input_token_cost_above_1hr",
    "cache_creation_input_token_cost_above_1hr",
    "search_context_cost_per_query",
    "code_interpreter_cost_per_session",
    "file_search_cost_per_gb_per_day",
    "vector_store_cost_per_gb_per_day",
    "output_cost_per_row",
    "output_cost_per_patch",
    "max_input_tokens",
    "max_output_tokens",
    "max_tokens",
    "mode",
    "litellm_provider",
    "output_vector_size",
    "supported_regions",
    "deprecation_date",
    "prompt_cache_min_tokens",
}


def to_litellm_entry(model: ModelData, key: str) -> dict[str, Any]:
    """Convert a merged ModelData into a LiteLLM cost-map style entry."""
    entry: dict[str, Any] = {}

    # Pricing: prefer explicit merged pricing, then pull others from capabilities
    pricing = model.pricing or {}
    for k in LITELLM_KEYS:
        if k in pricing and pricing[k] is not None:
            entry[k] = pricing[k]

    # Context
    if model.max_input_tokens:
        entry["max_input_tokens"] = model.max_input_tokens
    if model.max_output_tokens:
        entry["max_output_tokens"] = model.max_output_tokens
    mt = model.max_output_tokens or model.max_input_tokens
    if mt and "max_tokens" not in entry:
        # Legacy param: set to max_output if present else max_input
        entry["max_tokens"] = model.max_output_tokens or model.max_input_tokens

    entry["mode"] = model.mode or "chat"

    provider = model.litellm_provider or model.provider
    if provider:
        entry["litellm_provider"] = _to_litellm_provider(provider, model)

    # Capabilities -> supports_* flags
    for k, v in sorted(model.capabilities.items()):
        if k.startswith("supports_") and isinstance(v, bool):
            entry[k] = v

    # Inherit extra upstream LiteLLM fields not in merged pricing
    raw = model.sources.get("litellm_upstream") or {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k == "litellm_key":
                continue
            if k.startswith("supports_") and isinstance(v, bool):
                entry.setdefault(k, v)
            if k in LITELLM_KEYS and k not in entry and v is not None:
                entry[k] = v
    return entry


def to_litellm_cost_map(
    models: list[ModelData],
    include_user_only: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build a full LiteLLM model_prices_and_context_window.json map.

    When include_user_only is True, only models exposed on the user's LiteLLM
    proxy / OpenWebUI instances are included.

    Each merged logical model emits:
      - its canonical slug with merged pricing, AND
      - every variant id that carries its own per-token pricing (from the
        LiteLLM upstream map or CloudPrice flat map), preserving the
        provider/region cost-map granularity LiteLLM expects.
    """
    result: dict[str, dict[str, Any]] = {}
    for model in models:
        if include_user_only and not model.instances:
            continue
        result[model.canonical_slug] = to_litellm_entry(model, model.canonical_slug)
        for vid, row in (model.litellm_variants or {}).items():
            if vid == model.canonical_slug:
                continue
            result[vid] = dict(row)
            result[vid].setdefault("litellm_provider", model.litellm_provider or model.provider)
    return result


def _to_litellm_provider(provider: str, model: ModelData) -> str:
    """Map a generic provider name to a LiteLLM provider label."""
    mapping = {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "vertex_ai",
        "gemini": "vertex_ai",
        "meta-llama": "together_ai",
        "mistral": "mistral",
        "cohere": "cohere",
        "cohere.ai": "cohere",
        "deepseek": "deepseek",
        "x-ai": "x_ai",
        "groq": "groq",
        "azure": "azure",
        "azure_openai": "azure",
        "bedrock": "bedrock",
    }
    p = provider.lower()
    # Try direct mapping from known litellm providers in raw upstream source
    raw = model.sources.get("litellm_upstream") or {}
    if isinstance(raw, dict) and raw.get("litellm_provider"):
        return raw["litellm_provider"]
    return mapping.get(p, p)