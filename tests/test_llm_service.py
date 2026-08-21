"""
tests/test_llm_service.py
─────────────────────────
Unit tests for app/services/llm_service.py.

Strategy
────────
* The live Gemini API is never called.  All tests mock `_call_generate` —
  the named helper that wraps `client.models.generate_content` — so asyncio,
  thread-pool offloading, and Pydantic validation are all exercised while the
  network layer is fully isolated.

* Mocking `_call_generate` (rather than `client.models.generate_content`
  directly) is simpler and more robust: it avoids having to simulate the full
  `GenerateContentResponse` object and keeps tests decoupled from SDK internals.

* All async test functions are collected automatically by pytest-asyncio
  (asyncio_mode = auto in pytest.ini).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.models.resume import EvaluationResult, ParsedResume
from app.services.llm_service import evaluate_candidate_fit, extract_structured_resume


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_parsed_resume_json() -> str:
    """Valid JSON string that satisfies the ParsedResume schema."""
    return json.dumps(
        {
            "full_name": "Alice Chen",
            "email": "alice@example.com",
            "phone": "+1-555-0101",
            "skills": ["Python", "FastAPI", "MongoDB", "Docker"],
            "experience": [
                {
                    "company": "Tech Corp",
                    "role": "Backend Engineer",
                    "duration": "2021-2024",
                    "highlights": ["Built REST APIs", "Reduced latency by 35%"],
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
            "summary": "Experienced backend engineer with 3 years of FastAPI expertise.",
        }
    )


@pytest.fixture()
def valid_evaluation_result_json() -> str:
    """Valid JSON string that satisfies the EvaluationResult schema."""
    return json.dumps(
        {
            "match_score": 8.5,
            "justification": "Strong alignment with required Python and FastAPI skills.",
            "matched_skills": ["Python", "FastAPI", "MongoDB"],
            "missing_skills": ["Kubernetes"],
            "recommendation": "Strong Match",
        }
    )


@pytest.fixture()
def sample_parsed_resume() -> ParsedResume:
    """A ready-made ParsedResume instance for use in evaluation tests."""
    return ParsedResume(
        full_name="Alice Chen",
        email="alice@example.com",
        skills=["Python", "FastAPI", "MongoDB"],
    )


# ---------------------------------------------------------------------------
# Test: extract_structured_resume
# ---------------------------------------------------------------------------


class TestExtractStructuredResume:

    @pytest.mark.asyncio
    async def test_extract_structured_resume_success(
        self, valid_parsed_resume_json: str
    ) -> None:
        """
        When _call_generate returns a valid ParsedResume JSON string, the
        function must return a correctly populated ParsedResume instance.
        """
        with patch(
            "app.services.llm_service._call_generate",
            return_value=valid_parsed_resume_json,
        ):
            result = await extract_structured_resume("Alice Chen - Python Engineer...")

        assert isinstance(result, ParsedResume)
        assert result.full_name == "Alice Chen"
        assert result.email == "alice@example.com"
        assert "Python" in result.skills
        assert len(result.experience) == 1
        assert result.experience[0].company == "Tech Corp"
        assert len(result.education) == 1
        assert result.education[0].institution == "Stanford University"

    @pytest.mark.asyncio
    async def test_extract_structured_resume_passes_raw_text(self) -> None:
        """
        The raw_text argument must be forwarded as the `contents` parameter
        to _call_generate (verified by inspecting mock call args).
        """
        raw = "Jane Doe — Senior SWE — jane@example.com"
        minimal_json = json.dumps({"full_name": "Jane Doe"})

        with patch(
            "app.services.llm_service._call_generate",
            return_value=minimal_json,
        ) as mock_call:
            await extract_structured_resume(raw)

        # First positional arg to _call_generate is `contents`
        call_args = mock_call.call_args
        assert call_args.args[0] == raw

    @pytest.mark.asyncio
    async def test_extract_structured_resume_api_failure_raises_502(self) -> None:
        """
        Any exception from _call_generate must be caught and re-raised as
        HTTPException 502 — never propagated as a raw SDK error.
        """
        with patch(
            "app.services.llm_service._call_generate",
            side_effect=Exception("Simulated Gemini API timeout"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await extract_structured_resume("Some resume text")

        assert exc_info.value.status_code == 502
        assert exc_info.value.detail == "LLM Provider Error"

    @pytest.mark.asyncio
    async def test_extract_structured_resume_invalid_json_raises_502(self) -> None:
        """
        If the model returns malformed JSON, model_validate_json raises a
        ValidationError which must be wrapped as HTTPException 502.
        """
        with patch(
            "app.services.llm_service._call_generate",
            return_value="{not valid json !!!}",
        ):
            with pytest.raises(HTTPException) as exc_info:
                await extract_structured_resume("Some resume text")

        assert exc_info.value.status_code == 502


# ---------------------------------------------------------------------------
# Test: evaluate_candidate_fit
# ---------------------------------------------------------------------------


class TestEvaluateCandidateFit:

    @pytest.mark.asyncio
    async def test_evaluate_candidate_fit_success(
        self,
        sample_parsed_resume: ParsedResume,
        valid_evaluation_result_json: str,
    ) -> None:
        """
        When _call_generate returns a valid EvaluationResult JSON string, the
        function must return a correctly populated EvaluationResult instance.
        """
        job_desc = "Looking for a Python backend engineer with FastAPI experience."

        with patch(
            "app.services.llm_service._call_generate",
            return_value=valid_evaluation_result_json,
        ):
            result = await evaluate_candidate_fit(sample_parsed_resume, job_desc)

        assert isinstance(result, EvaluationResult)
        assert result.match_score == 8.5
        assert result.recommendation == "Strong Match"
        assert "Python" in result.matched_skills
        assert "Kubernetes" in result.missing_skills
        assert "FastAPI" in result.justification

    @pytest.mark.asyncio
    async def test_evaluate_candidate_fit_score_boundary(
        self, sample_parsed_resume: ParsedResume
    ) -> None:
        """match_score at exact boundary values (1.0 and 10.0) must parse cleanly."""
        for score, recommendation in [(1.0, "Not a Fit"), (10.0, "Strong Match")]:
            boundary_json = json.dumps(
                {
                    "match_score": score,
                    "justification": "Boundary test.",
                    "matched_skills": [],
                    "missing_skills": [],
                    "recommendation": recommendation,
                }
            )
            with patch(
                "app.services.llm_service._call_generate",
                return_value=boundary_json,
            ):
                result = await evaluate_candidate_fit(sample_parsed_resume, "JD text")

            assert result.match_score == score

    @pytest.mark.asyncio
    async def test_evaluate_candidate_fit_user_message_contains_resume_and_jd(
        self, sample_parsed_resume: ParsedResume, valid_evaluation_result_json: str
    ) -> None:
        """
        The user message passed to _call_generate must include both the
        serialised resume JSON and the job description text.
        """
        job_desc = "Unique job description marker XYZ-9999"

        with patch(
            "app.services.llm_service._call_generate",
            return_value=valid_evaluation_result_json,
        ) as mock_call:
            await evaluate_candidate_fit(sample_parsed_resume, job_desc)

        user_message: str = mock_call.call_args.args[0]
        assert "Alice Chen" in user_message          # resume content present
        assert "XYZ-9999" in user_message            # job description present

    @pytest.mark.asyncio
    async def test_llm_service_handles_api_failure(
        self, sample_parsed_resume: ParsedResume
    ) -> None:
        """
        Any exception from _call_generate during evaluation must be caught
        and re-raised as HTTPException 502.
        """
        with patch(
            "app.services.llm_service._call_generate",
            side_effect=Exception("Quota exceeded"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await evaluate_candidate_fit(sample_parsed_resume, "Some JD")

        assert exc_info.value.status_code == 502
        assert exc_info.value.detail == "LLM Provider Error"

    @pytest.mark.asyncio
    async def test_evaluate_candidate_fit_invalid_json_raises_502(
        self, sample_parsed_resume: ParsedResume
    ) -> None:
        """Malformed JSON response from the model must surface as 502."""
        with patch(
            "app.services.llm_service._call_generate",
            return_value="<html>Error page</html>",
        ):
            with pytest.raises(HTTPException) as exc_info:
                await evaluate_candidate_fit(sample_parsed_resume, "JD")

        assert exc_info.value.status_code == 502
