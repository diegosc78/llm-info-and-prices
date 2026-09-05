from __future__ import annotations

import logging

from ..models import SourcePayload
from .base import BaseFetcher

logger = logging.getLogger(__name__)


class LiteLLMUserFetcher(BaseFetcher):
    """Fetches the user's own LiteLLM proxy model list.

    Tries /v1/model/info (rich) first, then /v1/models (OpenAI-compatible),
    then /v1/model_group/info. Falls back gracefully.
    """

    name = "litellm_user"

    async def _fetch(self) -> SourcePayload:
        base = (self.settings.litellm_api_url or "").rstrip("/")
        if not base:
            raise ValueError("LiteLLM_USER_API_URL not configured")

        headers: dict[str, str] = {}
        if self.settings.litellm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.litellm_api_key}"

        items: dict[str, dict] = {}
        raw: dict = {}

        # Try rich /v1/model/info
        info_error: Exception | None = None
        try:
            resp = await self._request(
                "GET", f"{base}/v1/model/info", headers=headers
            )
            body = resp.json()
            raw["model_info"] = body
            data = body.get("data", []) if isinstance(body, dict) else []
            for item in data:
                if not isinstance(item, dict):
                    continue
                mi = item.get("model_info") or {}
                name = (
                    mi.get("internal_name")
                    or item.get("model_name")
                    or mi.get("model_name")
                )
                if not name:
                    continue
                entry = dict(item)
                entry["litellm_proxy_model_name"] = name
                # The actual upstream model id behind the public proxy name.
                lp = item.get("litellm_params") or {}
                origin = lp.get("model") if isinstance(lp, dict) else None
                if isinstance(origin, str) and origin and origin.lower() != name.lower():
                    entry["origin_model_id"] = origin
                items[name] = entry
        except Exception as exc:  # noqa: BLE001
            info_error = exc
            logger.debug("litellm_user /v1/model/info failed: %s", exc)

        # Fallback: OpenAI-compatible /v1/models
        if not items:
            err2: Exception | None = None
            try:
                resp = await self._request(
                    "GET", f"{base}/v1/models", headers=headers
                )
                body = resp.json()
                raw["models"] = body
                data = body.get("data", []) if isinstance(body, dict) else []
                for m in data:
                    mid = m.get("id")
                    if mid:
                        items[mid] = {"openai_compatible_id": mid, **m}
            except Exception as exc:  # noqa: BLE001
                err2 = exc

            if not items:
                # Try /v1/model_group/info
                try:
                    resp = await self._request(
                        "GET", f"{base}/v1/model_group/info", headers=headers
                    )
                    body = resp.json()
                    raw["model_group_info"] = body
                    data = body.get("data", []) if isinstance(body, dict) else []
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("model_name") or item.get("id")
                        if name:
                            items[name] = item
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(f"LiteLLM user proxy: no model source worked "
                                     f"(model_info: {info_error}, models: {err2}, "
                                     f"group: {exc})") from exc

        if not items:
            raise ValueError("LiteLLM user proxy: no models found")

        return SourcePayload(name=self.name, raw=raw, items=items)