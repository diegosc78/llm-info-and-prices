from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ..models import ModelData
from ..state import AppState
from .litellm_format import to_litellm_cost_map, to_litellm_entry

router = APIRouter()


def _state(request: Request) -> AppState:
    return request.app.state.data


def _serialize(model: ModelData, include_raw: bool = True) -> dict[str, Any]:
    """Serialize a merged model without runtime __prio leak."""
    pricing_src = dict(model.pricing_source)
    pricing = dict(model.pricing)
    return {
        "canonical_slug": model.canonical_slug,
        "provider": model.provider,
        "display_name": model.display_name,
        "description": model.description,
        "mode": model.mode,
        "context_length": model.context_length,
        "max_input_tokens": model.max_input_tokens,
        "max_output_tokens": model.max_output_tokens,
        "capabilities": dict(model.capabilities),
        "pricing": pricing,
        "pricing_source": pricing_src,
        "base_model_id": model.base_model_id,
        "resolved_price_from": model.resolved_price_from,
        "resolved_context_from": model.resolved_context_from,
        "urls": dict(model.urls),
        "aliases": dict(model.aliases),
        "instances": list(model.instances),
        "sources": dict(model.sources) if include_raw else None,
    }


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    state = _state(request)
    resp = {
        "status": "ok",
        "timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "total_models": state.total_models,
        "sources": [s.model_dump() for s in state.sources.values()],
    }
    ok_count = sum(1 for s in state.sources.values() if s.ok)
    if ok_count == 0:
        resp["status"] = "degraded"
    return resp


@router.get("/sources")
async def sources(request: Request) -> dict[str, Any]:
    state = _state(request)
    return {"sources": [s.model_dump() for s in state.sources.values()]}


@router.get("/models")
async def list_models(
    request: Request,
    q: str | None = Query(default=None, description="Free-text search"),
    provider: str | None = Query(default=None),
    mode: str | None = Query(default=None),
    user_instances: bool | None = Query(default=None, description="Only models in your LiteLLM/OpenWebUI"),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    state = _state(request)
    matched = state.search(q=q, provider=provider, mode=mode, user_instances=user_instances)
    total = len(matched)
    start = (page - 1) * per_page
    items = [_serialize(m, include_raw=False) for m in matched[start : start + per_page]]
    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "items": items,
    }


@router.get("/model")
async def get_model_by_query(
    request: Request, slug: str = Query(description="Model slug or alias id")
) -> dict[str, Any]:
    """Lookup a model by slug/alias via query param (avoids slash-in-path issues)."""
    state = _state(request)
    model = state.get_model(slug)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{slug}' not found")
    return _serialize(model, include_raw=True)


@router.get("/models/{slug:path}/litellm")
async def get_model_litellm(request: Request, slug: str) -> dict[str, Any]:
    state = _state(request)
    model = state.get_model(slug)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{slug}' not found")
    return to_litellm_entry(model, slug)


@router.get("/models/{slug:path}")
async def get_model(request: Request, slug: str) -> dict[str, Any]:
    state = _state(request)
    model = state.get_model(slug)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{slug}' not found")
    return _serialize(model, include_raw=True)


@router.get("/litellm-cost-map")
async def litellm_cost_map(
    request: Request,
    user_only: bool = Query(
        default=False,
        description="Only include models exposed on your LiteLLM proxy or OpenWebUI",
    ),
) -> dict[str, dict[str, Any]]:
    """Serve a LiteLLM model_prices_and_context_window.json-compatible cost map.

    LiteLLM validation: the map must contain at least MIN_MODEL_COUNT entries
    (default 50) and not shrink more than the backup map by MAX_SHRINK_RATIO.
    With user_only=True, ensure the map never drops below the minimum count by
    falling back to the full merged set when needed.
    """
    state = _state(request)
    model_map = to_litellm_cost_map(state.models, include_user_only=user_only)

    if user_only:
        min_count = state.settings.model_cost_map_min_model_count
        if len(model_map) < min_count:
            # Pad with the full merged map so LiteLLM validation passes.
            full = to_litellm_cost_map(state.models, include_user_only=False)
            for k, v in full.items():
                model_map.setdefault(k, v)

    return JSONResponse(model_map)


@router.get("/litellm-proxy")
async def litellm_proxy_config(
    request: Request,
    hostname: str = Query(default="https://litellm.example.com", description="Your LiteLLM proxy base URL"),
) -> dict[str, Any]:
    """Generate the LITELLM_MODEL_COST_MAP_URL / config snippet."""
    state = _state(request)
    url = f"{hostname.rstrip('/')}/litellm-cost-map"
    return {
        "LITELLM_MODEL_COST_MAP_URL": url,
        "instructions": (
            "Set this env var on your LiteLLM proxy before startup. "
            "Verify with 'curl {host}/model/cost_map/source'."
        ).format(host=hostname.rstrip("/")),
    }


@router.post("/refresh")
async def refresh(request: Request) -> dict[str, Any]:
    state = _state(request)
    statuses = await state.refresh()
    return {
        "status": "refreshed",
        "total_models": state.total_models,
        "sources": [s.model_dump() for s in statuses.values()],
    }