"""
tests/test_main.py
──────────────────
Integration smoke-tests for the FastAPI application entry-point.

Scope
─────
These tests exercise the real ASGI app (not a mock) through HTTPX's
ASGITransport.  They validate that routes are reachable and return the correct
HTTP status codes and response shapes — catching wiring issues before Phase 2
routes are added.
"""

from httpx import Client


class TestHealthCheck:
    """Group: /health liveness probe."""

    def test_health_check_status_200(self, client: Client) -> None:
        """GET /health must return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. Body: {response.text}"
        )

    def test_health_check_response_body(self, client: Client) -> None:
        """GET /health must return JSON body {"status": "healthy"}."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data, "Response JSON is missing 'status' key"
        assert data["status"] == "healthy", (
            f"Expected 'healthy', got {data['status']!r}"
        )

    def test_health_check_content_type(self, client: Client) -> None:
        """GET /health must respond with application/json content type."""
        response = client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")


class TestNotFound:
    """Group: generic 404 behaviour."""

    def test_unknown_route_returns_404(self, client: Client) -> None:
        """Any unregistered route must return HTTP 404."""
        response = client.get("/this-route-does-not-exist")
        assert response.status_code == 404
