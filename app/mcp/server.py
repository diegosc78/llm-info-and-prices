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
    return "\n".join(parts)


def create_mcp(state: AppState) -> FastMCP:
    wrapper = ModelInfoMCP(state)
    return wrapper.mcp