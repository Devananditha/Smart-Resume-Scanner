"""
tests/test_main.py
──────────────────
Integration smoke-tests for the FastAPI application entry-point.

Scope
─────
These tests exercise the real ASGI app (not a mock) through HTTPX's
ASGITransport.  They validate that routes are reachable and return the correct
HTTP status codes and response shapes.
"""

from unittest.mock import AsyncMock, patch

from httpx import Client

from app.models.resume import EvaluationResult, ParsedResume


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


class TestScreeningEndpoints:
    """Group: /api/v1/screen and /api/v1/candidates endpoints."""

    @patch("app.main.extract_text_from_pdf")
    @patch("app.main.extract_structured_resume")
    @patch("app.main.evaluate_candidate_fit")
    @patch("app.main.save_candidate_evaluation")
    def test_screen_candidate_success(
        self, mock_save, mock_eval, mock_extract_struct, mock_extract_pdf, client: Client
    ):
        # Mock responses
        mock_extract_pdf.return_value = "Mock PDF text"
        mock_parsed = ParsedResume(
            full_name="Alice",
            email="alice@example.com",
            skills=["Python"],
            experience=[],
            education=[],
        )
        mock_extract_struct.return_value = mock_parsed
        mock_evaluation = EvaluationResult(
            match_score=9.0,
            justification="Great match",
            matched_skills=["Python"],
            missing_skills=[],
            recommendation="Strong Match",
        )
        mock_eval.return_value = mock_evaluation
        mock_save.return_value = "uuid-1234"

        # Make the request
        files = {"file": ("resume.pdf", b"dummy pdf content", "application/pdf")}
        data = {"job_description": "Need a Python dev"}
        response = client.post("/api/v1/screen", files=files, data=data)

        # Assert response
        assert response.status_code == 200
        json_resp = response.json()
        assert json_resp["candidate_id"] == "uuid-1234"
        assert json_resp["parsed_resume"]["full_name"] == "Alice"
        assert json_resp["evaluation"]["match_score"] == 9.0

        # Assert mocks called
        mock_extract_pdf.assert_called_once()
        mock_extract_struct.assert_called_once_with("Mock PDF text")
        mock_eval.assert_called_once_with(mock_parsed, "Need a Python dev")
        mock_save.assert_called_once_with(
            job_description="Need a Python dev",
            parsed=mock_parsed,
            eval_result=mock_evaluation
        )

    def test_screen_candidate_invalid_file_type(self, client: Client):
        files = {"file": ("image.png", b"dummy image", "image/png")}
        data = {"job_description": "Need a Python dev"}
        response = client.post("/api/v1/screen", files=files, data=data)

        assert response.status_code == 400
        assert "Invalid file type" in response.text

    @patch("app.main.list_shortlisted_candidates")
    def test_get_shortlisted_candidates(self, mock_list, client: Client):
        mock_list.return_value = [
            {"id": "uuid-1", "evaluation": {"match_score": 9.5}},
            {"id": "uuid-2", "evaluation": {"match_score": 8.0}},
        ]

        response = client.get("/api/v1/candidates?min_score=8.0")
        assert response.status_code == 200
        json_resp = response.json()
        assert len(json_resp) == 2
        assert json_resp[0]["id"] == "uuid-1"
        mock_list.assert_called_once_with(8.0)
