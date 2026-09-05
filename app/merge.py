from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from .models import ModelData, SourcePayload

logger = logging.getLogger(__name__)

# Price priority: higher index wins.
PRICE_PRIORITY = ["litellm_upstream", "portkey", "cloudprice", "openrouter"]

SOURCE_LABEL = {
    "litellm_upstream": "LiteLLM cost map",
    "portkey": "Portkey",
    "cloudprice": "CloudPrice",
    "openrouter": "OpenRouter",
    "livebench": "LiveBench",
    "litellm_user": "NUESTROS modelos LiteLLM",
    "openwebui": "NUESTROS modelos OpenWebUI",
}

_USER_SOURCES = {"litellm_user", "openwebui"}


def normalize_id(value: str) -> str:
    """Normalize a model identifier to a comparable form.

    - strips whitespace/lowercases
    - unifies separators . and - (only for matching) into /
      BUT keep '.' and '-' where they matter (e.g. gpt-4o-2024-08-06)
    """
    value = (value or "").strip().strip('"').lower()
    return value


def suffix_id(value: str) -> str:
    """Return the last path segment of an identifier (model part)."""
    v = normalize_id(value)
    if "/" in v:
        return v.rsplit("/", 1)[-1]
    return v


def provider_from_id(value: str, source: str | None = None) -> str:
    v = normalize_id(value)
    if "/" in v:
        return v.split("/", 1)[0]
    return source or ""


class MergeEngine:
    """De-duplicates model variants across sources and merges their data."""

    def __init__(self, payloads: list[SourcePayload]):
        self.payloads = payloads

    def build(self) -> list[ModelData]:
        # ----- Pass 1: gather items with their owning source -----
        # items: list of (source_name, variant_id, entry_dict, keyset)
        items: list[tuple[str, str, dict, set[str]]] = []

        # alias authority from CloudPrice: alias -> canonical
        cloudprice_alias_map: dict[str, str] = {}
        for payload in self.payloads:
            for vid, entry in payload.items.items():
                aliases = entry.get("cloudprice_aliases") or []
                if aliases and payload.name == "cloudprice":
                    canonical = entry.get("cloudprice_key") or vid
                    for a in aliases:
                        cloudprice_alias_map[normalize_id(str(a))] = canonical

        for payload in self.payloads:
            for vid, entry in payload.items.items():
                entry = dict(entry)
                keys = self._matching_keys(payload.name, vid, entry)
                items.append((payload.name, vid, entry, keys))

        # ----- Pass 2: union-find clusters based on shared keys -----
        parent: dict[int, int] = {}
        key_to_idx: dict[str, int] = defaultdict(list)

        def find(x: int) -> int:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for idx, (_, _, _, keys) in enumerate(items):
            parent[idx] = idx
            for key in keys:
                for other in key_to_idx.get(key, []):
                    union(idx, other)
                key_to_idx[key].append(idx)

        clusters: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(items)):
            clusters[find(idx)].append(idx)

        # ----- Pass 3: build ModelData per cluster -----
        result: list[ModelData] = []
        for cluster in clusters.values():
            model = self._merge_cluster(cluster, items, cloudprice_alias_map)
            if model is not None:
                result.append(model)

        # ----- Pass 4: attach benchmark scores (LiveBench) by id matching -----
        self._attach_benchmarks(result, self.payloads)

        logger.info("merge: %d clusters from %d items", len(result), len(items))
        return result

    def _matching_keys(
        self, source: str, vid: str, entry: dict
    ) -> set[str]:
        keys: set[str] = set()
        vid = normalize_id(vid)
        keys.add(vid)  # exact

        # Bare IDs (no provider prefix) like LiteLLM keys ("gpt-4o-mini") or
        # Portkey model names bridge to provider-prefixed variants from other
        # sources. Provider-prefixed IDs rely on exact/alias matches instead,
        # so openai/gpt-4o and groq/gpt-4o are never merged by name alone.
        is_catalog = (
            source == "cloudprice" and entry.get("cloudprice_source") == "models_catalog"
        )
        if "/" not in vid:
            keys.add(suffix_id(vid))

        # OpenRouter exposes canonical_slug and id
        for field in ("id", "canonical_slug"):
            val = entry.get(field)
            if isinstance(val, str) and val:
                keys.add(normalize_id(val))

        # CloudPrice *catalog* ids array is the authoritative alias bridge.
        # Flat (litellm_model_prices) entries are provider-specific pricing
        # rows and should only match by their exact id — not drag all the
        # regional/provider variants of the same logical model together.
        if is_catalog:
            for a in entry.get("cloudprice_aliases") or []:
                keys.add(normalize_id(str(a)))

        # Portkey entries are keyed portkey/{provider}/{model}; expose the
        # provider/model form as well (matches OpenRouter/CloudPrice ids)
        if source == "portkey":
            provider = entry.get("portkey_provider")
            model = entry.get("portkey_model")
            if provider and model:
                keys.add(normalize_id(model))
                keys.add(f"{normalize_id(provider)}/{normalize_id(model)}")

        return keys

    def _merge_cluster(
        self,
        cluster: list[int],
        items: list[tuple[str, str, dict, set[str]]],
        cloudprice_alias_map: dict[str, str],
    ) -> ModelData | None:
        cluster_items = [items[i] for i in cluster]
        if not cluster_items:
            return None

        # Canonical slug selection
        slug, provider = self._pick_slug(cluster_items, cloudprice_alias_map)

        model = ModelData(canonical_slug=slug, provider=provider)

        # Track all variant ids & sources
        for source, vid, entry, _ in cluster_items:
            model.aliases[vid] = source
            model.sources.setdefault(source, {})[vid] = entry

        # Merge raw per-source data
        for source in ("openrouter", "cloudprice", "portkey", "litellm_user", "openwebui"):
            source_map: dict[str, Any] = {}
            for s, vid, entry, _ in cluster_items:
                if s == source:
                    source_map[vid] = entry
            if source_map:
                model.sources[source] = source_map if len(source_map) > 1 else next(
                    iter(source_map.values())
                )

        # ----- Merge identity / description fields -----
        for source, vid, entry, _ in cluster_items:
            if source == "openrouter":
                model.provider = provider or entry.get("id", "").split("/")[0]
                model.display_name = entry.get("name") or model.display_name
                model.description = entry.get("description") or model.description
            elif source == "cloudprice":
                model.display_name = (
                    entry.get("display_name") or model.display_name
                )
                model.description = entry.get("description") or model.description
                model.provider = (
                    entry.get("creator") or model.provider
                )
            elif source == "litellm_user":
                d = entry.get("litellm_params") or {}
                lt_model = d.get("model") or d.get("model_name") or entry.get(
                    "model_name"
                )
                if lt_model:
                    model.aliases.setdefault(str(lt_model), source)
                mi = entry.get("model_info") or {}
                if mi.get("litellm_provider"):
                    model.litellm_provider = mi["litellm_provider"]
                model.instances.append("litellm_user")
            elif source == "openwebui":
                info = entry.get("info") or {}
                meta = info.get("meta") or {}
                name = info.get("name") or entry.get("name")
                desc = meta.get("description") or entry.get("description")
                if name:
                    model.display_name = name or model.display_name
                if desc:
                    model.description = desc or model.description
                model.instances.append("openwebui")

        if not model.provider:
            model.provider = provider_from_id(slug, "unknown")

        # ----- Base / origin model capture (OpenWebUI wins, else LiteLLM) -----
        for source, vid, entry, _ in cluster_items:
            if source == "openwebui":
                base = entry.get("base_model_id")
                if (
                    isinstance(base, str)
                    and base
                    and normalize_id(base) != normalize_id(vid)
                ):
                    model.base_model_id = base
            elif source == "litellm_user":
                origin = entry.get("origin_model_id")
                if (
                    model.base_model_id is None
                    and isinstance(origin, str)
                    and origin
                    and normalize_id(origin) != normalize_id(vid)
                ):
                    model.base_model_id = origin

        # ----- Merge context windows (take richest but not smaller than others) -----
        max_input = model.max_input_tokens
        max_output = model.max_output_tokens
        ctx = model.context_length
        for source, vid, entry, _ in cluster_items:
            if source == "openrouter":
                c = entry.get("context_length")
                if isinstance(c, int) and c > 0:
                    ctx = max(ctx or 0, c)
                tp = entry.get("top_provider") or {}
                if isinstance(tp.get("max_completion_tokens"), int):
                    max_output = max(max_output or 0, tp["max_completion_tokens"])
            elif source == "cloudprice":
                c = entry.get("context_window")
                if isinstance(c, int) and c > 0:
                    ctx = max(ctx or 0, c)
                mi = entry.get("max_input_tokens")
                mo = entry.get("max_output_tokens")
                if isinstance(mi, int) and mi > 0:
                    max_input = max(max_input or 0, mi)
                if isinstance(mo, int) and mo > 0:
                    max_output = max(max_output or 0, mo)
            elif source == "litellm_upstream":
                mi = entry.get("max_input_tokens")
                mo = entry.get("max_output_tokens")
                mt = entry.get("max_tokens")
                if isinstance(mi, int) and mi > 0:
                    max_input = max(max_input or 0, mi)
                if isinstance(mo, int) and mo > 0:
                    max_output = max(max_output or 0, mo)
                elif isinstance(mt, int) and mt > 0:
                    max_output = max(max_output or 0, mt)
                c = entry.get("context_length")
                if isinstance(c, int) and c > 0:
                    ctx = max(ctx or 0, c)

        model.context_length = ctx or max_input
        model.max_input_tokens = max_input or ctx
        model.max_output_tokens = max_output

        # ----- Merge capability flags (union) -----
        caps: dict[str, bool] = {}
        for source, vid, entry, _ in cluster_items:
            if source == "litellm_upstream":
                for k, v in entry.items():
                    if k.startswith("supports_") and isinstance(v, bool):
                        caps.setdefault(k, False)
                        caps[k] = caps[k] or v
            elif source == "cloudprice":
                cap_map = entry.get("capabilities")
                if isinstance(cap_map, dict):
                    for k, v in cap_map.items():
                        bk = f"supports_{k}"
                        caps.setdefault(bk, False)
                        caps[bk] = caps[bk] or bool(v)
            elif source == "openrouter":
                arch = entry.get("architecture") or {}
                in_mods = arch.get("input_modalities") or []
                out_mods = arch.get("output_modalities") or []
                if "image" in in_mods:
                    caps["supports_vision"] = True
                if "audio" in in_mods:
                    caps["supports_audio_input"] = True
                if "audio" in out_mods:
                    caps["supports_audio_output"] = True
                if "tools" in (entry.get("supported_parameters") or []):
                    caps["supports_function_calling"] = True
                if "structured_outputs" in (entry.get("supported_parameters") or []):
                    caps["supports_response_schema"] = True
        model.capabilities = dict(sorted(caps.items()))

        # ----- Merge pricing (priority: openrouter > cloudprice > ...) -----
        pricing: dict[str, Any] = {}
        pricing_source: dict[str, str] = {}
        self._merge_pricing(cluster_items, pricing, pricing_source, model)
        model.pricing = {k: v for k, v in pricing.items() if not k.endswith("__prio")}
        model.pricing_source = {
            k: v for k, v in pricing_source.items() if not k.endswith("__prio")
        }
        model.litellm_variants = self._collect_litellm_variants(cluster_items)
        model.mode = self._pick_mode(cluster_items, model)

        # ----- URLs / docs -----
        for source, vid, entry, _ in cluster_items:
            if source == "openrouter":
                links = entry.get("links") or {}
                if links.get("details"):
                    model.urls["openrouter_details"] = (
                        f"https://openrouter.ai{links['details']}"
                    )
            elif source == "cloudprice":
                model.urls["cloudprice_model"] = (
                    f"https://cloudprice.net/models/{entry.get('id', vid)}"
                )
            elif source == "litellm_upstream":
                src = entry.get("source")
                if isinstance(src, str) and src:
                    model.urls["litellm_source"] = src

        return model

    def _pick_slug(
        self,
        cluster_items: list[tuple[str, str, dict, set[str]]],
        cloudprice_alias_map: dict[str, str],
    ) -> tuple[str, str]:
        # 1) OpenRouter canonical_slug / id (cleanest: provider/model)
        for source, _, entry, _ in cluster_items:
            if source == "openrouter" and (entry.get("canonical_slug") or entry.get("id")):
                cs = entry.get("canonical_slug") or entry.get("id")
                return normalize_id(cs), cs.split("/")[0]
        # 2) CloudPrice canonical id (the model catalog `id` field)
        for source, _, entry, _ in cluster_items:
            if source == "cloudprice":
                cid = entry.get("id")
                if isinstance(cid, str) and cid:
                    return normalize_id(cid), entry.get("creator") or cid.split("-")[0]
                cid = entry.get("cloudprice_key")
                if isinstance(cid, str) and cid:
                    # check the alias map for a canonical target
                    target = cloudprice_alias_map.get(normalize_id(cid))
                    if target and target != cid:
                        return normalize_id(target), target.split("-")[0]
                    return normalize_id(cid), cid.split("-")[0]
        # 3) The variant id that looks like provider/model with a real provider
        for source, vid, _, _ in cluster_items:
            if "/" in vid:
                return normalize_id(vid), vid.split("/")[0]
        # 4) any first
        source, vid, _, _ = cluster_items[0]
        return normalize_id(vid), provider_from_id(vid, source)

    def _merge_pricing(
        self,
        cluster_items: list[tuple[str, str, dict, set[str]]],
        pricing: dict[str, Any],
        pricing_source: dict[str, str],
        model: ModelData,
    ) -> None:
        """Apply pricing from all sources; higher priority wins per key."""

        def _take(prices: dict[str, Any], src_prio: int, src_label: str) -> None:
            for k, v in prices.items():
                if v is None:
                    continue
                cur_prio = pricing_source.get(k)
                priority = PRICE_PRIORITY.index(src_label) if src_label in PRICE_PRIORITY else -1
                if priority >= 0 and priority > pricing_source.get(k + "__prio", -1):
                    pricing[k] = v
                    pricing_source[k] = src_label
                    pricing_source[k + "__prio"] = priority

        # OpenRouter (strings like "0.00003" -> float per token)
        for source, _, entry, _ in cluster_items:
            if source != "openrouter":
                continue
            pr = entry.get("pricing")
            if not isinstance(pr, dict):
                continue
            prices: dict[str, float] = {}
            for k in ("prompt", "completion"):
                v = pr.get(k)
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                prices[k] = f
            prices = {
                "input_cost_per_token": prices.get("prompt"),
                "output_cost_per_token": prices.get("completion"),
            }
            prices = {k: v for k, v in prices.items() if v is not None}
            _take(prices, 3, "openrouter")

        # CloudPrice (flat map is per-token; catalog summary is per 1M)
        for source, _, entry, _ in cluster_items:
            if source != "cloudprice":
                continue
            src = entry.get("cloudprice_source")
            prices: dict[str, float] = {}
            if src == "litellm_model_prices":
                in_c = entry.get("input_cost_per_token")
                out_c = entry.get("output_cost_per_token")
                cache_r = entry.get("cache_read_input_token_cost")
                cache_w = entry.get("cache_creation_input_token_cost")
                for name, v in (
                    ("input_cost_per_token", in_c),
                    ("output_cost_per_token", out_c),
                    ("cache_read_input_token_cost", cache_r),
                    ("cache_creation_input_token_cost", cache_w),
                ):
                    try:
                        f = float(v)
                    except (TypeError, ValueError):
                        continue
                    prices[name] = f
            elif src == "models_catalog" and isinstance(entry.get("pricing"), dict):
                summary = entry["pricing"].get("summary") or {}
                # summary reports min and max per-1M across providers; take the
                # max for conservative (worst-case) cost estimation.
                in_v = summary.get("max_input_per_1m", summary.get("min_input_per_1m"))
                out_v = summary.get(
                    "max_output_per_1m", summary.get("min_output_per_1m")
                )
                try:
                    prices["input_cost_per_token"] = float(in_v) / 1e6
                except (TypeError, ValueError):
                    pass
                try:
                    prices["output_cost_per_token"] = float(out_v) / 1e6
                except (TypeError, ValueError):
                    pass
            _take(prices, 2, "cloudprice")

        # Portkey (cents/token -> divide by 100 for $/token)
        for source, _, entry, _ in cluster_items:
            if source != "portkey":
                continue
            pc = entry.get("portkey_pricing") or {}
            if not isinstance(pc, dict):
                continue
            pc = pc.get("pricing_config") or {}
            if not isinstance(pc, dict):
                continue
            payg = pc.get("pay_as_you_go") or {}
            prices: dict[str, float] = {}
            for name in ("request_token", "response_token"):
                price = (payg.get(name) or {}).get("price")
                try:
                    f = float(price) / 100.0
                except (TypeError, ValueError):
                    continue
                key = (
                    "input_cost_per_token"
                    if name == "request_token"
                    else "output_cost_per_token"
                )
                prices[key] = f
            _take(prices, 1, "portkey")

        # LiteLLM upstream (already per-token)
        for source, _, entry, _ in cluster_items:
            if source != "litellm_upstream":
                continue
            prices: dict[str, Any] = {}
            for k in (
                "input_cost_per_token",
                "output_cost_per_token",
                "input_cost_per_audio_token",
                "output_cost_per_audio_token",
                "cache_read_input_token_cost",
                "cache_creation_input_token_cost",
                "input_cost_per_image",
                "output_cost_per_image",
                "input_cost_per_pixel",
                "output_cost_per_pixel",
                "input_cost_per_video_per_second",
                "input_cost_per_audio_per_second",
                "input_cost_per_query",
                "output_cost_per_query",
                "input_cost_per_1k_tokens_cache_hit",
            ):
                v = entry.get(k)
                if v is not None:
                    prices[k] = v
            _take(prices, 0, "litellm_upstream")

    def _collect_litellm_variants(
        self, cluster_items: list[tuple[str, str, dict, set[str]]]
    ) -> dict[str, dict[str, Any]]:
        """Per-variant LiteLLM-style rows for ids that carry their own pricing.

        Keeps the provider/region granularity visible in the cost map even when
        several variants were merged into a single logical model.
        """
        variants: dict[str, dict[str, Any]] = {}
        for source, vid, entry, _ in cluster_items:
            if source == "litellm_upstream":
                row = {k: v for k, v in entry.items() if k != "litellm_key"}
                row["mode"] = row.get("mode", "chat")
                variants[vid] = row
            elif source == "cloudprice" and entry.get("cloudprice_source") == "litellm_model_prices":
                keys = (
                    "litellm_provider",
                    "mode",
                    "input_cost_per_token",
                    "output_cost_per_token",
                    "cache_read_input_token_cost",
                    "cache_creation_input_token_cost",
                    "max_input_tokens",
                    "max_output_tokens",
                )
                row = {k: entry[k] for k in keys if k in entry}
                row["mode"] = row.get("mode", "chat")
                if entry.get("litellm_provider"):
                    row["litellm_provider"] = entry["litellm_provider"]
                variants[self._variant_key(vid)] = row
        # the canonical slug entry shouldn't be duplicated as a variant
        return variants

    def _variant_key(self, vid: str) -> str:
        return vid

    def _pick_mode(
        self,
        cluster_items: list[tuple[str, str, dict, set[str]]],
        model: ModelData,
    ) -> str:
        priorities = ["litellm_upstream", "cloudprice", "portkey", "openrouter"]
        for source in priorities:
            for s, _, entry, _ in cluster_items:
                if s != source:
                    continue
                if source == "litellm_upstream":
                    m = entry.get("mode")
                    if m:
                        return m
                if source == "cloudprice":
                    m = entry.get("type") or ""
                    if isinstance(m, dict):
                        m = m.get("primary") or ""
                    if m:
                        return self._map_mode(m)
                    src = entry.get("cloudprice_source")
                    if src == "litellm_model_prices":
                        m = entry.get("mode")
                        if m:
                            return m
                if source == "portkey":
                    g = entry.get("portkey_general") or {}
                    if not isinstance(g, dict):
                        continue
                    t = g.get("type")
                    if isinstance(t, dict) and t.get("primary"):
                        return self._map_mode(t["primary"])
                if source == "openrouter":
                    arch = entry.get("architecture") or {}
                    m = arch.get("modality") or ""
                    if m:
                        out = m.split("->")[-1] if "->" in m else m
                        if out in ("text", "image", "audio", "video"):
                            return "chat" if out == "text" else out
                        return self._map_mode(out)
        return model.mode

    @staticmethod
    def _map_mode(m: str) -> str:
        """Normalize to LiteLLM mode vocabulary."""
        m = (m or "").lower()
        mapping = {
            "chat": "chat",
            "text": "chat",
            "completion": "completion",
            "embedding": "embedding",
            "embeddings": "embedding",
            "image": "image_generation",
            "image_generation": "image_generation",
            "audio": "audio_transcription",
            "speech": "audio_speech",
            "transcription": "audio_transcription",
            "tts": "audio_speech",
            "moderation": "moderation",
            "rerank": "rerank",
            "ocr": "chat",
            "video": "chat",
        }
        return mapping.get(m, m)

    # ------------------------------------------------------------------
    # Benchmark (LiveBench) attachment
    # ------------------------------------------------------------------
    def _attach_benchmarks(
        self,
        models: list[ModelData],
        payloads: list[SourcePayload],
    ) -> None:
        """Attach LiveBench scores to merged models by fuzzy id matching.

        LiveBench ids (e.g. claude-opus-4-5-20251101-thinking-64k-high-effort)
        never match a canonical slug exactly, so they are token-normalized and
        matched against each model's slug/alias tokens (stem first).
        """
        lb_payload = next((p for p in payloads if p.name == "livebench"), None)
        if lb_payload is None or not lb_payload.items:
            return
        rows = [
            {**row, "_tokens": _clean_tokens(row.get("livebench_id") or "")}
            for row in lb_payload.items.values()
        ]
        matched = 0
        for model in models:
            row = _best_livebench_match(model, rows)
            if row is None:
                continue
            score = dict(row)
            score.pop("_tokens", None)
            model.benchmarks["livebench"] = score
            matched += 1
        logger.info("merge: livebench scores attached to %d/%d models",
                    matched, len(models))


_STOP_TOKENS = {
    "latest", "preview", "thinking", "reasoning", "effort",
    "high", "medium", "low", "xhigh", "xlow", "auto",
    "beta", "experimental", "api", "chat", "instruct", "finetune", "turbo",
}
_DATE_RE = re.compile(r"\d{4,8}\b")
_SIZE_RE = re.compile(r"\d+[kmb]\b")
_ISO_DATE_RE = re.compile(r"-\d{8}\b|-\d{4}-\d{2}-\d{2}\b")


def _clean_tokens(v: str) -> tuple[frozenset[str], tuple[str, ...]]:
    """Normalize a model id to a comparable signature for benchmark matching.

    Signature = (letter tokens as a set, version digits as an ORDERED tuple).
    Letters are order-free so claude-opus-4-5 matches claude-4-5-opus, while
    version digits keep their order so gpt-4.5 is never confused with gpt-5.4
    and gpt-5.5 stays distinct from gpt-5. Deployment suffixes (chat/preview/
    thinking/high-effort/dates) are dropped as noise.
    """
    v = (v or "").lower()
    v = _ISO_DATE_RE.sub("", v)  # drop -2025-12-11 / -20251101 chunks before tokenizing
    letters: set[str] = set()
    digits: list[str] = []
    for t in re.findall(r"[a-z0-9]+", v):
        if t in _STOP_TOKENS:
            continue
        if _DATE_RE.fullmatch(t):
            continue
        if _SIZE_RE.fullmatch(t):
            continue
        if t.isdigit():
            digits.append(t)
        else:
            letters.add(t)
    return frozenset(letters), tuple(digits)


def _best_livebench_match(
    model: ModelData, rows: list[dict]
) -> dict | None:
    stems: list[tuple] = []
    fulls: list[tuple] = []
    for key in [model.canonical_slug, model.canonical_slug.replace("/", "-"), *model.aliases.keys()]:
        if not key:
            continue
        seg = key.split("/")[-1]
        fulls.append(_clean_tokens(key))
        stems.append(_clean_tokens(seg))

    best: dict | None = None
    best_key: tuple = (0, "")
    for row in rows:
        c = row["_tokens"]
        if not c or not any(c):
            continue
        # Only exact signature equality against the canonical slug / its stem
        # or any alias stem/full id. Subset/"family" matching is deliberately
        # avoided: it would mislabel variants (codex/nano/...) with the family
        # score, and unordered token sets would confuse gpt-5.4 with gpt-4.5.
        if not (any(c == s for s in stems) or any(c == f for f in fulls)):
            continue
        key = (len(c[0]) + len(c[1]), row.get("livebench_id") or "")
        if key > best_key:
            best_key = key
            best = row
    return best