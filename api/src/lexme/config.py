"""Application settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration for the API."""

    database_url: str
    tei_url: str = "http://tei:80"
    frontend_origin: str = "http://localhost:5173"
    gemini_api_key: str = ""
    openrouter_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
