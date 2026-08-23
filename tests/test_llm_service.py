"""
tests/test_llm_service.py
─────────────────────────
Unit tests for app/services/llm_service.py (single-pass architecture).

Strategy
────────
* The live Groq and Gemini APIs are never called. Tests mock
  ``_call_groq_single_pass`` and ``_call_gemini_single_pass`` — the named
  synchronous helpers — so asyncio offloading and Pydantic validation are
  fully exercised while the network layer is isolated.

* Tests cover:
  1. Happy-path Groq success returning a valid CandidateAnalysis.
  2. Groq failure → Gemini fallback activates and succeeds.
  3. Both providers fail → HTTPException 502.
  4. Invalid JSON from either provider → HTTPException 502.
  5. User message forwarding verified via mock call args.

* All async functions are collected by pytest-asyncio (asyncio_mode=auto).
"""

from __future__ import annotations

import json
from unittest.mock import patch

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


# ---------------------------------------------------------------------------
# Tests: analyze_candidate_single_pass
# ---------------------------------------------------------------------------


class TestAnalyzeCandidateSinglePass:

    RESUME = "Alice Chen — Python Engineer — alice@example.com"
    JD     = "Looking for a Python FastAPI backend engineer."

    @pytest.mark.asyncio
    async def test_groq_success_returns_candidate_analysis(
        self, valid_analysis_json: str
    ) -> None:
        """
        When _call_groq_single_pass succeeds, the function must return a
        fully validated CandidateAnalysis with correct nested fields.
        """
        with patch(
            "app.services.llm_service._call_groq_single_pass",
            return_value=valid_analysis_json,
        ):
            result = await analyze_candidate_single_pass(self.RESUME, self.JD)

        assert isinstance(result, CandidateAnalysis)
        assert isinstance(result.parsed_resume, ParsedResume)
        assert isinstance(result.evaluation, EvaluationResult)
        assert result.parsed_resume.full_name == "Alice Chen"
        assert result.evaluation.match_score == 8.5
        assert result.evaluation.recommendation == "Strong Match"
        assert "Python" in result.parsed_resume.skills

    @pytest.mark.asyncio
    async def test_user_message_contains_resume_and_jd(
        self, valid_analysis_json: str
    ) -> None:
        """
        The user_message forwarded to _call_groq_single_pass must include
        both the resume text and the job description text.
        """
        with patch(
            "app.services.llm_service._call_groq_single_pass",
            return_value=valid_analysis_json,
        ) as mock_call:
            await analyze_candidate_single_pass(self.RESUME, self.JD)

        user_msg: str = mock_call.call_args.args[0]
        assert "Alice Chen" in user_msg          # resume content present
        assert "FastAPI backend engineer" in user_msg  # JD content present

    @pytest.mark.asyncio
    async def test_groq_fails_gemini_succeeds(
        self, valid_analysis_json: str
    ) -> None:
        """
        When Groq raises any exception, _call_gemini_single_pass must be
        invoked and the function must still return a valid CandidateAnalysis.
        """
        with (
            patch(
                "app.services.llm_service._call_groq_single_pass",
                side_effect=Exception("429 Groq rate limit"),
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

    @pytest.mark.asyncio
    async def test_both_providers_fail_raises_502(self) -> None:
        """
        If both Groq and Gemini raise exceptions, HTTPException 502 must
        be raised — raw provider errors must never propagate.
        """
        with (
            patch(
                "app.services.llm_service._call_groq_single_pass",
                side_effect=Exception("Groq overloaded"),
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

    @pytest.mark.asyncio
    async def test_invalid_json_from_groq_falls_back_to_gemini(
        self, valid_analysis_json: str
    ) -> None:
        """
        If Groq returns malformed JSON that fails Pydantic validation,
        the Gemini fallback should be triggered and succeed.
        """
        with (
            patch(
                "app.services.llm_service._call_groq_single_pass",
                return_value="{not valid json !!!}",
            ),
            patch(
                "app.services.llm_service._call_gemini_single_pass",
                return_value=valid_analysis_json,
            ) as mock_gemini,
        ):
            result = await analyze_candidate_single_pass(self.RESUME, self.JD)

        assert isinstance(result, CandidateAnalysis)
        mock_gemini.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_json_from_both_providers_raises_502(self) -> None:
        """
        Malformed JSON from both providers must ultimately surface as
        HTTPException 502 — never as a raw ValidationError or JSONDecodeError.
        """
        with (
            patch(
                "app.services.llm_service._call_groq_single_pass",
                return_value="<html>error</html>",
            ),
            patch(
                "app.services.llm_service._call_gemini_single_pass",
                return_value="<html>gemini error</html>",
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await analyze_candidate_single_pass(self.RESUME, self.JD)

        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_score_boundary_values_parse_correctly(
        self, valid_analysis_json: str
    ) -> None:
        """
        match_score at exact boundary values (1.0 and 10.0) must validate
        without error through the CandidateAnalysis model.
        """
        for score, rec in [(1.0, "Not a Fit"), (10.0, "Strong Match")]:
            data = json.loads(valid_analysis_json)
            data["evaluation"]["match_score"] = score
            data["evaluation"]["recommendation"] = rec
            boundary_json = json.dumps(data)

            with patch(
                "app.services.llm_service._call_groq_single_pass",
                return_value=boundary_json,
            ):
                result = await analyze_candidate_single_pass(self.RESUME, self.JD)

            assert result.evaluation.match_score == score
