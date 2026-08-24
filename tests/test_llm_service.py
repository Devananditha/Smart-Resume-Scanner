"""
tests/test_llm_service.py
─────────────────────────
Unit tests for app/services/llm_service.py (single-pass architecture).

Strategy
────────
* The live Groq and Gemini APIs are never called. Tests mock
  ``groq_client.chat.completions.create`` and ``_call_gemini_single_pass``
  so asyncio offloading and Pydantic validation are fully exercised while
  the network layer is isolated.

* Tests cover:
  1. Happy-path Groq success returning a valid CandidateAnalysis.
  2. Groq failure → Gemini fallback activates and succeeds.
  3. Both providers fail → HTTPException 502.
  4. Invalid JSON from either provider → HTTPException 502.

* All async functions are collected by pytest-asyncio (asyncio_mode=auto).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.models.resume import CandidateAnalysis, EvaluationResult, ParsedResume
from app.services.llm_service import analyze_candidate_single_pass


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_analysis_json() -> str:
    """Valid JSON string satisfying the CandidateAnalysis combined schema."""
    return json.dumps(
        {
            "parsed_resume": {
                "full_name": "Alice Chen",
                "email": "alice@example.com",
                "phone": "+1-555-0101",
                "skills": ["Python", "FastAPI", "MongoDB", "Docker"],
                "experience": [
                    {
                        "company": "Tech Corp",
                        "role": "Backend Engineer",
                        "duration": "2021-2024",
                        "highlights": ["Built REST APIs", "Reduced latency 35%"],
                    }
                ],
                "education": [
                    {
                        "institution": "Stanford University",
                        "degree": "B.Sc. Computer Science",
                        "graduation_year": "2021",
                        "gpa": "3.9/4.0",
                    }
                ],
                "summary": "Experienced backend engineer.",
            },
            "evaluation": {
                "match_score": 8.5,
                "justification": "Strong alignment with Python and FastAPI skills.",
                "matched_skills": ["Python", "FastAPI", "MongoDB"],
                "missing_skills": ["Kubernetes"],
                "recommendation": "Strong Match",
            },
        }
    )


def mock_groq_response(json_string: str) -> MagicMock:
    """Helper to generate a mock Groq completions response."""
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json_string
    mock_resp.choices = [mock_choice]
    return mock_resp


# ---------------------------------------------------------------------------
# Tests: analyze_candidate_single_pass
# ---------------------------------------------------------------------------


class TestAnalyzeCandidateSinglePass:

    RESUME = "Alice Chen — Python Engineer — alice@example.com"
    JD     = "Looking for a Python FastAPI backend engineer."

    @pytest.mark.asyncio
    @patch("app.services.llm_service.asyncio.sleep", new_callable=AsyncMock)
    async def test_groq_success_returns_candidate_analysis(
        self, mock_sleep: AsyncMock, valid_analysis_json: str
    ) -> None:
        """
        When Groq succeeds, the function must return a
        fully validated CandidateAnalysis with correct nested fields.
        """
        mock_resp = mock_groq_response(valid_analysis_json)
        with patch(
            "app.services.llm_service.groq_client.chat.completions.create",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await analyze_candidate_single_pass(self.RESUME, self.JD)

        assert isinstance(result, CandidateAnalysis)
        assert isinstance(result.parsed_resume, ParsedResume)
        assert isinstance(result.evaluation, EvaluationResult)
        assert result.parsed_resume.full_name == "Alice Chen"
        assert result.evaluation.match_score == 8.5
        assert result.evaluation.recommendation == "Strong Match"
        assert "Python" in result.parsed_resume.skills
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.services.llm_service.asyncio.sleep", new_callable=AsyncMock)
    async def test_groq_fails_gemini_succeeds(
        self, mock_sleep: AsyncMock, valid_analysis_json: str
    ) -> None:
        """
        When Groq raises any exception, it should sleep 1s, then invoke
        _call_gemini_single_pass and still return a valid CandidateAnalysis.
        """
        with (
            patch(
                "app.services.llm_service.groq_client.chat.completions.create",
                new_callable=AsyncMock,
                side_effect=Exception("Groq error"),
            ),
            patch(
                "app.services.llm_service._call_gemini_single_pass",
                return_value=valid_analysis_json,
            ) as mock_gemini,
        ):
            result = await analyze_candidate_single_pass(self.RESUME, self.JD)

        assert isinstance(result, CandidateAnalysis)
        assert result.parsed_resume.full_name == "Alice Chen"
        mock_gemini.assert_called_once()
        mock_sleep.assert_called_once_with(1)

    @pytest.mark.asyncio
    @patch("app.services.llm_service.asyncio.sleep", new_callable=AsyncMock)
    async def test_both_providers_fail_raises_502(self, mock_sleep: AsyncMock) -> None:
        """
        If both Groq and Gemini raise exceptions, HTTPException 502 must
        be raised — raw provider errors must never propagate.
        """
        with (
            patch(
                "app.services.llm_service.groq_client.chat.completions.create",
                new_callable=AsyncMock,
                side_effect=Exception("Groq rate limited"),
            ),
            patch(
                "app.services.llm_service._call_gemini_single_pass",
                side_effect=Exception("Gemini quota exceeded"),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await analyze_candidate_single_pass(self.RESUME, self.JD)

        assert exc_info.value.status_code == 502
        assert exc_info.value.detail == "LLM Provider Error"
        mock_sleep.assert_called_once_with(1)

    @pytest.mark.asyncio
    @patch("app.services.llm_service.asyncio.sleep", new_callable=AsyncMock)
    async def test_invalid_json_from_groq_falls_back_to_gemini(
        self, mock_sleep: AsyncMock, valid_analysis_json: str
    ) -> None:
        """
        If Groq returns malformed JSON that fails Pydantic validation,
        the Gemini fallback should be triggered and succeed.
        """
        mock_resp = mock_groq_response("{not valid json !!!}")
        with (
            patch(
                "app.services.llm_service.groq_client.chat.completions.create",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ),
            patch(
                "app.services.llm_service._call_gemini_single_pass",
                return_value=valid_analysis_json,
            ) as mock_gemini,
        ):
            result = await analyze_candidate_single_pass(self.RESUME, self.JD)

        assert isinstance(result, CandidateAnalysis)
        mock_gemini.assert_called_once()
        # Invalid JSON causes an exception during model_validate_json, so it hits the except block
        mock_sleep.assert_called_once_with(1)

    @pytest.mark.asyncio
    @patch("app.services.llm_service.asyncio.sleep", new_callable=AsyncMock)
    async def test_invalid_json_from_both_providers_raises_502(self, mock_sleep: AsyncMock) -> None:
        """
        Malformed JSON from both providers must ultimately surface as
        HTTPException 502 — never as a raw ValidationError or JSONDecodeError.
        """
        mock_resp = mock_groq_response("<html>error</html>")
        with (
            patch(
                "app.services.llm_service.groq_client.chat.completions.create",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ),
            patch(
                "app.services.llm_service._call_gemini_single_pass",
                return_value="<html>gemini error</html>",
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await analyze_candidate_single_pass(self.RESUME, self.JD)

        assert exc_info.value.status_code == 502
        mock_sleep.assert_called_once_with(1)
