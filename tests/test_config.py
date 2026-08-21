"""
tests/test_config.py
────────────────────
Unit tests for app/config.py — validates that Settings loads correctly from
environment variables and that all required fields resolve to non-empty values.

Strategy
────────
`monkeypatch.setenv` injects fake env vars *before* the Settings object is
constructed inside the test, bypassing any real .env file so the suite is
fully self-contained and safe to run in CI without secrets.
"""

import importlib

import pytest


class TestSettingsLoad:
    """Group: settings initialisation from environment variables."""

    def test_settings_load_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Settings resolves `gemini_api_key` and `mongodb_url` from env vars.
        The test injects synthetic values so it never depends on a real .env file.
        """
        # Arrange — inject required env vars before importing Settings
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-123")
        monkeypatch.setenv("MONGODB_URL", "mongodb://localhost:27017")
        monkeypatch.setenv("DATABASE_NAME", "test_resume_screener")

        # Re-import config with the patched environment
        import app.config as config_module
        importlib.reload(config_module)
        settings = config_module.Settings()  # type: ignore[call-arg]

        # Assert — all required fields are present and match injected values
        assert settings.gemini_api_key is not None, "gemini_api_key must not be None"
        assert settings.gemini_api_key != "", "gemini_api_key must not be empty"
        assert settings.mongodb_url is not None, "mongodb_url must not be None"
        assert settings.mongodb_url.startswith("mongodb://"), (
            "mongodb_url must be a valid MongoDB connection string"
        )

    def test_database_name_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DATABASE_NAME falls back to 'resume_screener' when not set."""
        monkeypatch.setenv("GEMINI_API_KEY", "any-key")
        monkeypatch.delenv("DATABASE_NAME", raising=False)

        import app.config as config_module
        importlib.reload(config_module)
        settings = config_module.Settings()  # type: ignore[call-arg]

        assert settings.database_name == "resume_screener"

    def test_mongodb_url_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MONGODB_URL falls back to 'mongodb://localhost:27017' when not set."""
        monkeypatch.setenv("GEMINI_API_KEY", "any-key")
        monkeypatch.delenv("MONGODB_URL", raising=False)

        import app.config as config_module
        importlib.reload(config_module)
        settings = config_module.Settings()  # type: ignore[call-arg]

        assert settings.mongodb_url == "mongodb://localhost:27017"
