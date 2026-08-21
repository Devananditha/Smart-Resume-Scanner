"""
tests/conftest.py
─────────────────
Session-scoped fixtures shared across the entire test suite.

Notes
─────
* `TestClient` (from Starlette, re-exported by FastAPI) is the correct
  synchronous wrapper for ASGI apps — it handles event-loop lifecycle and
  lifespan events internally.
* `httpx.ASGITransport` only supports the *async* context-manager protocol, so
  it cannot be used with the synchronous `httpx.Client` directly.
"""

from collections.abc import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app as _app


# ── App fixture ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def test_app() -> FastAPI:
    """Returns the FastAPI application instance under test."""
    return _app


# ── Sync HTTP client ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def client(test_app: FastAPI) -> Generator[TestClient, None, None]:
    """
    Yields a synchronous Starlette TestClient wrapping the FastAPI app.
    Session-scoped to avoid re-creating the client for every test.
    """
    with TestClient(app=test_app, raise_server_exceptions=True) as c:
        yield c
