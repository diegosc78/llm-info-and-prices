from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..state import AppState

NAME = "llm-info-and-prices"


class ModelInfoMCP:
    """FastMCP HTTP server exposing model catalog data to agents."""

    def __init__(self, state: AppState):
        self.state = state
        self.mcp = FastMCP(NAME)

        @self.mcp.tool()
        def list_models(
            provider: str | None = None,
            mode: str | None = None,
            user_instances: bool | None = None,
        ) -> str:
            """List available AI models. Returns name, canonical slug, provider and mode for each model."""
            models = state.search(provider=provider, mode=mode, user_instances=user_instances)
            lines = []
            for m in models[:500]:
                lines.append(
                    f"- {m.display_name or m.canonical_slug} | slug={m.canonical_slug}"
                    f" | provider={m.provider} | mode={m.mode}"
                    f" | instances={','.join(m.instances) or 'n/a'}"
                )
            return "\n".join(lines) if lines else "No models found."

        @self.mcp.tool()
        def search_models(query: str) -> str:
            """Search models by free-text query (matches name, description, provider, aliases, URLs). Returns matching model slugs."""
            matches = state.search(q=query)
            lines = []
            for m in matches[:200]:
                lines.append(
                    f"- {m.display_name or m.canonical_slug} | slug={m.canonical_slug}"
                    f" | provider={m.provider} | mode={m.mode}"
                )
            return "\n".join(lines) if lines else "No models matched the query."

        @self.mcp.tool()
        def get_model_details(slug: str) -> str:
            """Get full details (prices, context windows, capabilities, URLs, aliases, description) for a model by its slug or any alias."""
            model = state.get_model(slug)
            if model is None:
                return f"Model '{slug}' not found."
            return _format_model(model)

        @self.mcp.tool()
        def top_models(
            top: int = 10,
            axis: str = "agentic_coding",
            min_score: float | None = None,
            min_context_tokens: int | None = None,
            out_price_per_1m: float | None = None,
            capabilities: list[str] | None = None,
            sort_by: str = "score",
            dedupe: bool = False,
        ) -> str:
            """Recommend models for a task by LiveBench benchmark score with optional filters.

            axis: overall, coding, agentic_coding, reasoning, math, data_analysis,
            language, instruction_following. Filters: min_score (0-100),
            min_context_tokens (window), out_price_per_1m (max USD / 1M output
            tokens), capabilities (e.g. ['function_calling','structured_outputs']),
            sort_by: score|price|value, dedupe: collapse provider variants.
            """
            cap_list = [] if capabilities is None else [
                c if c.startswith("supports_") else f"supports_{c}" for c in capabilities
            ]
            rows: list[tuple] = []
            for m in state.models:
                lb = (m.benchmarks or {}).get("livebench") or {}
                score = lb.get(axis)
                if score is None:
                    continue
                if min_score is not None and score < min_score:
                    continue
                if cap_list and any(not m.capabilities.get(c) for c in cap_list):
                    continue
                if min_context_tokens is not None and (
                    (m.max_input_tokens or m.context_length or 0) < min_context_tokens
                ):
                    continue
                out_1m = None
                v = m.pricing.get("output_cost_per_token")
                if isinstance(v, (int, float)):
                    out_1m = v * 1_000_000
                if out_price_per_1m is not None and (out_1m is None or out_1m > out_price_per_1m):
                    continue
                value = (score / out_1m) if out_1m else None
                rows.append((m, round(score, 1), out_1m, value, lb.get("livebench_id", "")))

            if sort_by == "price":
                rows.sort(key=lambda r: (r[2] is None, r[2] or float("inf")))
            elif sort_by == "value":
                rows.sort(key=lambda r: (r[3] is None, -(r[3] or 0)))
            else:
                rows.sort(key=lambda r: (-r[1], r[2] or float("inf")))

            if dedupe:
                seen: set[str] = set()
                kept: list = []
                for r in rows:
                    if r[4] in seen:
                        continue
                    seen.add(r[4])
                    kept.append(r)
                rows = kept

            lines = [f"Top {top} by {axis} (sort={sort_by}):"]
            for m, score, out_1m, value, lbid in rows[:top]:
                price = f"${out_1m:.3f}" if out_1m is not None else "n/a"
                val = f" value={value:.1f}" if value is not None else ""
                ctx = m.max_input_tokens or m.context_length
                lines.append(
                    f"- {m.canonical_slug} | {axis}={score} | out$/1M={price}{val}"
                    f" | ctx={ctx}"
                )
            return "\n".join(lines)

    def format_model(self, model) -> str:
        return _format_model(model)


def _format_model(model) -> str:
    parts = [
        f"Canonical slug: {model.canonical_slug}",
        f"Provider: {model.provider}",
        f"Display name: {model.display_name or 'n/a'}",
        f"Mode: {model.mode}",
        f"Context length: {model.context_length or 'n/a'}",
        f"Max input tokens: {model.max_input_tokens or 'n/a'}",
        f"Max output tokens: {model.max_output_tokens or 'n/a'}",
    ]
    if model.description:
        parts.append(f"Description: {model.description}")
    if model.pricing:
        price_lines = []
        for k, v in sorted(model.pricing.items()):
            src = model.pricing_source.get(k, "?")
            price_lines.append(f"  {k} = {v} (from {src})")
        parts.append("Pricing:\n" + "\n".join(price_lines))
    if model.context_length or model.max_input_tokens or model.max_output_tokens:
        win = [
            f"context_length={model.context_length}"
            if model.context_length else None,
            f"max_input_tokens={model.max_input_tokens}"
            if model.max_input_tokens else None,
            f"max_output_tokens={model.max_output_tokens}"
            if model.max_output_tokens else None,
        ]
        parts.append("Context window: " + ", ".join(w for w in win if w))
    if model.capabilities:
        caps = ", ".join(
            f"{k.replace('supports_', '')}={str(v).lower()}"
            for k, v in sorted(model.capabilities.items())
        )
        parts.append(f"Capabilities: {caps}")
    if model.urls:
        parts.append("URLs:\n" + "\n".join(f"  {k}: {v}" for k, v in model.urls.items()))
    if model.aliases:
        parts.append("Aliases:\n" + "\n".join(f"  {a} ({s})" for a, s in sorted(model.aliases.items())))
    if model.instances:
        parts.append("Available on your instances: " + ", ".join(model.instances))
    if model.base_model_id:
        parts.append("Base / origin model: " + model.base_model_id)
    if model.resolved_price_from:
        parts.append("Pricing derived from: " + model.resolved_price_from)
    if model.resolved_context_from:
        parts.append("Context window derived from: " + model.resolved_context_from)
    if model.benchmarks.get("livebench"):
        lb = model.benchmarks["livebench"]
        parts.append(
            "LiveBench ("
            + lb.get("table", "?")
            + "): overall="
            + _fmt_score(lb.get("overall"))
            + ", agentic_coding="
            + _fmt_score(lb.get("agentic_coding"))
            + ", coding="
            + _fmt_score(lb.get("coding"))
            + ", reasoning="
            + _fmt_score(lb.get("reasoning"))
        )
    return "\n".join(parts)


def _fmt_score(v) -> str:
    return "n/a" if v is None else f"{v:g}"


def create_mcp(state: AppState) -> FastMCP:
    wrapper = ModelInfoMCP(state)
    return wrapper.mcp