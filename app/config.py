from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "LLM Info & Prices"
    port: int = 8000
    log_level: str = "INFO"

    # ---- User instances ----
    litellm_api_url: str | None = None
    litellm_api_key: str | None = None

    openwebui_api_url: str | None = None
    openwebui_api_key: str | None = None

    # ---- OpenRouter ----
    openrouter_api_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str | None = None

    # ---- Refresh / cache ----
    refresh_interval_seconds: int = 3600
    cache_ttl_seconds: int = 3600
    http_timeout_seconds: float = 15.0
    http_max_retries: int = 3
    startup_refresh: bool = True

    # ---- LiteLLM cost map validation ----
    model_cost_map_min_model_count: int = 50
    model_cost_map_max_shrink_ratio: float = 0.5


@lru_cache
def get_settings() -> Settings:
    return Settings()
