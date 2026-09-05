from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import router
from .config import get_settings
from .state import AppState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state = AppState(settings)
    if settings.startup_refresh:
        logger.info("initial refresh at startup...")
        await state.start()
    app.state.data = state

    from .mcp.server import create_mcp

    mcp = create_mcp(state)
    app.state.mcp = mcp
    mcp_app = mcp.http_app(path="/")
    app.mount("/mcp", mcp_app)
    try:
        async with mcp_app.lifespan(app):
            yield
    finally:
        await state.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Aggregates LLM model info & prices from LiteLLM, Portkey, CloudPrice, "
            "OpenRouter, your LiteLLM proxy and OpenWebUI. Serves a LiteLLM-compatible "
            "cost map, a REST API and an MCP SSE server."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)

    return app


app = create_app()