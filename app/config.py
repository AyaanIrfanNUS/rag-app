"""
Configurations for the application.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM configuration
    gemini_api_key: str
    primary_model: str = "gemini-3.5-flash"
    fallback_model: str = "gemini-3.5-flash"  # same model for now, can be changed later

    # Langsmith configuration
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "myFirstObs"
    langchain_tracing: bool = True

    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance, loaded once and reused everywhere."""
    return Settings()