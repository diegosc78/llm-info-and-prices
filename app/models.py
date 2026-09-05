from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SourceStatus(BaseModel):
    name: str
    ok: bool = False
    last_fetch: datetime | None = None
    models_count: int = 0
    error: str | None = None
    stale: bool = False


class SourcePayload(BaseModel):
    name: str
    raw: Any = None
    models: list[dict[str, Any]] = Field(default_factory=list)
    # map of variant id -> model dict, before canonicalization
    items: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ModelData(BaseModel):
    canonical_slug: str
    provider: str = ""
    display_name: str = ""
    description: str = ""
    mode: str = "chat"
    litellm_provider: str = ""
    context_length: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    # Merged capability flags (union across sources)
    capabilities: dict[str, bool] = Field(default_factory=dict)

    # Merged pricing (LiteLLM-style, $ per token/unit)
    pricing: dict[str, Any] = Field(default_factory=dict)

    # Price provenance: pricing key -> source name
    pricing_source: dict[str, str] = Field(default_factory=dict)

    # Per-variant LiteLLM-style pricing entries, keyed by the variant id.
    # e.g. azure/eu/gpt-4o... carries its own rates from the cloudprice flat map.
    litellm_variants: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # URL / docs metadata
    urls: dict[str, str] = Field(default_factory=dict)

    # Aliases: variant id -> source name it came from
    aliases: dict[str, str] = Field(default_factory=dict)

    # Which user instances expose this model
    instances: list[str] = Field(default_factory=list)

    # Raw source-specific payloads (kept verbatim for rich REST responses)
    sources: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sources: list[SourceStatus] = Field(default_factory=list)
    total_models: int = 0
