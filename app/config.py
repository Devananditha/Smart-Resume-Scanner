"""
app/config.py
─────────────
Centralised application settings loaded from environment variables (or a
.env file when present).  All other modules should import the singleton
`settings` object rather than reading os.environ directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration resolved from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # silently ignore unknown env vars
    )

    # ── Google Gemini ────────────────────────────────────────────────────────
    gemini_api_key: str

    # ── MongoDB ──────────────────────────────────────────────────────────────
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "resume_screener"


# Module-level singleton — import this everywhere.
settings = Settings()
