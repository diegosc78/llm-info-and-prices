from __future__ import annotations

import csv
import io
import logging
import re
from typing import Any

from ..models import SourcePayload
from .base import BaseFetcher

logger = logging.getLogger(__name__)

LIVEBENCH_BASE = "https://livebench.ai"

# Task -> LiveBench category. Category score = mean of its tasks.
CATEGORIES: dict[str, list[str]] = {
    "coding": ["code_completion", "code_generation"],
    "agentic_coding": ["javascript", "typescript", "python"],
    "reasoning": ["theory_of_mind", "zebra_puzzle", "spatial", "logic_with_navigation"],
    "math": ["AMPS_Hard", "integrals_with_game", "math_comp", "olympiad"],
    "data_analysis": ["consecutive_events", "tablejoin", "tablereformat"],
    "language": ["connections", "plot_unscrambling", "typos"],
    "instruction_following": ["paraphrase", "simplify", "story_generation", "summarize"],
}

# Fallback tables (newest first) used when GitHub discovery is unavailable.
_FALLBACK_TABLES = [
    "table_2026_06_25.csv",
    "table_2026_01_08.csv",
    "table_2025_12_23.csv",
    "table_2025_11_25.csv",
    "table_2025_05_30.csv",
    "table_2025_04_25.csv",
    "table_2025_04_02.csv",
    "table_2024_11_25.csv",
]

_TABLE_RE = re.compile(r"table_(\d{4})_(\d{2})_(\d{2})\.csv")


class LiveBenchFetcher(BaseFetcher):
    """LiveBench benchmark scores (contamination-free leaderboard).

    Downloads the monthly leaderboard CSV (all per-task scores), computes
    category averages (coding, agentic coding, reasoning, math, data analysis,
    language, instruction following) and an overall score per model.
    """

    name = "livebench"

    def __init__(self, settings=None):
        super().__init__(settings)
        self._table_url: str | None = None

    async def _fetch(self) -> SourcePayload:
        url = await self._resolve_table_url()
        resp = await self._request("GET", url)
        rows = self._parse_csv(resp.text)

        table = _TABLE_RE.search(url)
        table_label = (
            f"{table.group(2)}-{table.group(3)}-{table.group(1)}"
            if table
            else url.rsplit("/", 1)[-1]
        )

        items: dict[str, dict[str, Any]] = {}
        for r in rows:
            mid = (r.get("model") or "").strip()
            if not mid:
                continue
            tasks = {
                k: _to_float(v)
                for k, v in r.items()
                if k != "model" and v not in ("", None)
            }
            tasks = {k: v for k, v in tasks.items() if v is not None}
            cats = {cat: _mean(tasks.get(t) for t in tasks_of)
                    for cat, tasks_of in CATEGORIES.items()}
            overall = _mean(tasks.values())
            items[mid] = {
                "livebench_id": mid,
                "table": table_label,
                "url": url,
                "overall": overall,
                "tasks": tasks,
                **cats,
            }

        if not items:
            raise ValueError("LiveBench: no model scores parsed")
        logger.info("livebench: %d scored models (table %s)", len(items), table_label)
        return SourcePayload(name=self.name, raw={"url": url, "table": table_label}, items=items)

    async def _resolve_table_url(self) -> str:
        if self.settings.livebench_table_url:
            return self.settings.livebench_table_url
        # Prefer GitHub discovery for the newest monthly table.
        try:
            client = self._get_client()
            resp = await client.get(
                "https://api.github.com/repos/LiveBench/livebench.github.io/contents/public",
                timeout=self.settings.http_timeout_seconds,
            )
            if resp.status_code == 200:
                names = [
                    x.get("name", "")
                    for x in resp.json()
                    if isinstance(x, dict)
                ]
                tables = sorted(
                    (n for n in names if n.startswith("table_") and n.endswith(".csv")),
                    reverse=True,
                )
                if tables:
                    self._table_url = f"{LIVEBENCH_BASE}/{tables[0]}"
                    return self._table_url
        except Exception as exc:  # noqa: BLE001
            logger.warning("livebench: GitHub discovery failed (%s), using fallback table", exc)
        for name in _FALLBACK_TABLES:
            url = f"{LIVEBENCH_BASE}/{name}"
            try:
                resp = await self._get_client().get(
                    url, timeout=self.settings.http_timeout_seconds
                )
                if resp.status_code == 200:
                    return url
            except Exception:  # noqa: BLE001
                continue
        raise ValueError("LiveBench: no reachable table URL")

    @staticmethod
    def _parse_csv(text: str) -> list[dict[str, str]]:
        reader = csv.DictReader(io.StringIO(text))
        return [dict(row) for row in reader if dict(row)]


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(values: Any) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)  # type: ignore[arg-type]