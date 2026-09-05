from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ..models import ModelData
from ..state import AppState
from .litellm_format import to_litellm_cost_map, to_litellm_entry

router = APIRouter()

# LiveBench score axes exposed by /models/top
BENCH_AXES = (
    "overall",
    "coding",
    "agentic_coding",
    "reasoning",
    "math",
    "data_analysis",
    "language",
    "instruction_following",
)

TOP_SORTS = ("score", "price", "value")


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
        "benchmarks": dict(model.benchmarks),
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


@router.get("/models/top")
async def top_models(
    request: Request,
    top: int = Query(default=10, ge=1, le=200),
    axis: Literal[
        "overall",
        "coding",
        "agentic_coding",
        "reasoning",
        "math",
        "data_analysis",
        "language",
        "instruction_following",
    ] = Query(default="agentic_coding"),
    min_score: float | None = Query(default=None, ge=0, le=100),
    min_context_tokens: int | None = Query(default=None, ge=1),
    max_input_per_1m: float | None = Query(default=None, gt=0),
    max_output_per_1m: float | None = Query(default=None, gt=0),
    capabilities: list[str] = Query(
        default=[],
        description="Required capabilities (all must be true), e.g. capabilities=function_calling&capabilities=structured_outputs",
    ),
    sort_by: Literal["score", "price", "value"] = Query(default="score"),
    dedupe: Literal["none", "model"] = Query(
        default="none",
        description="'model' collapses regional/gateway variants that share the same underlying model, keeping the best",
    ),
    q: str | None = Query(default=None),
    provider: str | None = Query(default=None),
    mode: str | None = Query(default=None),
    user_instances: bool | None = Query(default=None),
) -> dict[str, Any]:
    """Top 'n' models by benchmark score within price/window/capability filters.

    Ranked selection for agentic coding etc.: filters by price range (per 1M
    tokens), minimum context window, required capabilities (function calling,
    structured outputs, ...) and minimum benchmark score; scores come from
    LiveBench (attached by model-id matching in the merge).
    """
    state = _state(request)
    matched = state.search(q=q, provider=provider, mode=mode, user_instances=user_instances)

    # Normalize capability names (accept with or without supports_ prefix)
    req_caps = [
        c if c.startswith("supports_") else f"supports_{c}"
        for c in capabilities
        if c
    ]

    rows: list[dict[str, Any]] = []
    for model in matched:
        lb = model.benchmarks.get("livebench") or {}
        score = lb.get(axis)
        if score is None:
            continue  # only models with the requested benchmark axis qualify
        if min_score is not None and score < min_score:
            continue
        if req_caps and any(not model.capabilities.get(c) for c in req_caps):
            continue
        ctx = model.max_input_tokens or model.context_length or 0
        if min_context_tokens is not None and ctx < min_context_tokens:
            continue
        in_1m, out_1m = _price_per_1m(model)
        if max_input_per_1m is not None and (in_1m is None or in_1m > max_input_per_1m):
            continue
        if max_output_per_1m is not None and (out_1m is None or out_1m > max_output_per_1m):
            continue

        # value = score per unit of cost (default: $/1M output tokens)
        base = out_1m if out_1m is not None else in_1m
        value = round(score / base, 3) if (score is not None and base) else None
        rows.append(
            {
                "model": _serialize(model, include_raw=False),
                "score": round(score, 1),
                "value_score": value,
                "price_per_1m": {"input": in_1m, "output": out_1m},
            }
        )

    if not rows:
        return _top_response(axis, sort_by, top, matched, filtered=0, items=[])

    if sort_by == "price":
        rows.sort(key=lambda r: (r["price_per_1m"]["output"] is None,
                                 r["price_per_1m"]["output"] or float("inf"),
                                 r["price_per_1m"]["input"] or float("inf")))
    elif sort_by == "value":
        rows.sort(key=lambda r: (r["value_score"] is None, -(r["value_score"] or 0)))
    else:  # score
        rows.sort(key=lambda r: (-r["score"],
                                 r["price_per_1m"]["output"] or float("inf")))

    if dedupe == "model":
        rows = _dedupe_by_model(rows)

    return _top_response(axis, sort_by, top, matched, filtered=len(rows), items=rows[:top])


def _dedupe_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse regional/gateway/user variants of the same benchmarked model.

    Every row in /models/top carries a LiveBench id (that's what qualified it);
    rows sharing that id are the same underlying model exposed under different
    provider/gateway/instance slugs. The top-ranked one is kept and the
    alias-slugs of the rest are merged into it.
    """
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        lb = row["model"].get("benchmarks", {}).get("livebench")
        key = (lb or {}).get("livebench_id")
        if not key:
            continue
        if key not in seen:
            seen[key] = row
        else:
            merged_aliases = seen[key]["model"].setdefault("aliases", {})
            for a, s in row["model"].get("aliases", {}).items():
                merged_aliases.setdefault(a, s)
    return list(seen.values())


def _top_response(
    axis: str,
    sort_by: str,
    top: int,
    matched: list[ModelData],
    filtered: int,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "axis": axis,
        "sort_by": sort_by,
        "top": top,
        "total_considered": len(matched),
        "total_qualified": filtered,
        "returned": len(items),
        "items": items,
    }


def _price_per_1m(model: ModelData) -> tuple[float | None, float | None]:
    def one(side: str) -> float | None:
        v = model.pricing.get(f"{side}_cost_per_token")
        if isinstance(v, (int, float)):
            return round(v * 1_000_000, 6)
        return None

    return one("input"), one("output")


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