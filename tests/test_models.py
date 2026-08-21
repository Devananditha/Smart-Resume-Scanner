"""
tests/test_models.py
────────────────────
Unit tests for all Pydantic schemas in app/models/resume.py.

Testing philosophy
──────────────────
* No mocking — tests exercise real Pydantic validation logic.
* Each test class maps 1-to-1 with a schema, making failures easy to triage.
* Invalid-data tests use `pytest.raises(ValidationError)` to assert both that
  an error is raised AND that it targets the correct field.
"""

import pytest
from pydantic import ValidationError

from app.models.resume import (
    CandidateRecord,
    Education,
    EvaluationResult,
    ParsedResume,
    WorkExperience,
)


# ── Education ─────────────────────────────────────────────────────────────────


class TestEducation:
    def test_valid_full(self) -> None:
        edu = Education(
            institution="MIT",
            degree="B.Sc. Computer Science",
            graduation_year="2022",
            gpa="3.9/4.0",
        )
        assert edu.institution == "MIT"
        assert edu.gpa == "3.9/4.0"

    def test_optional_fields_default_none(self) -> None:
        edu = Education(institution="Oxford", degree="M.Sc. AI")
        assert edu.graduation_year is None
        assert edu.gpa is None

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Education(institution="Harvard")  # type: ignore[call-arg]  # degree missing
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("degree",) for e in errors)


# ── WorkExperience ────────────────────────────────────────────────────────────


class TestWorkExperience:
    def test_valid_with_highlights(self) -> None:
        exp = WorkExperience(
            company="Google",
            role="Senior SWE",
            duration="Jan 2020 – Dec 2023",
            highlights=["Led team of 8", "Reduced latency by 40%"],
        )
        assert exp.company == "Google"
        assert len(exp.highlights) == 2

    def test_highlights_defaults_to_empty_list(self) -> None:
        exp = WorkExperience(company="Acme", role="Intern")
        assert exp.highlights == []

    def test_missing_company_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            WorkExperience(role="Analyst")  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("company",) for e in errors)


# ── ParsedResume ──────────────────────────────────────────────────────────────


class TestParsedResume:
    def test_parsed_resume_valid_data(self) -> None:
        resume = ParsedResume(
            full_name="Jane Doe",
            email="jane@example.com",
            phone="+1-555-0100",
            skills=["Python", "FastAPI", "MongoDB"],
            experience=[
                WorkExperience(
                    company="Startup Inc.",
                    role="Backend Engineer",
                    duration="2021–2023",
                    highlights=["Built REST APIs", "Deployed on AWS"],
                )
            ],
            education=[
                Education(
                    institution="Stanford",
                    degree="B.Sc. CS",
                    graduation_year="2021",
                )
            ],
            summary="Experienced backend engineer with 2+ years of FastAPI.",
        )
        assert resume.full_name == "Jane Doe"
        assert "Python" in resume.skills
        assert len(resume.experience) == 1
        assert resume.experience[0].company == "Startup Inc."

    def test_optional_fields_absent(self) -> None:
        resume = ParsedResume(full_name="John Smith")
        assert resume.email is None
        assert resume.phone is None
        assert resume.summary is None
        assert resume.skills == []
        assert resume.experience == []
        assert resume.education == []

    def test_missing_full_name_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ParsedResume()  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("full_name",) for e in errors)


# ── EvaluationResult ──────────────────────────────────────────────────────────


class TestEvaluationResult:
    def test_evaluation_result_valid(self) -> None:
        result = EvaluationResult(
            match_score=8.5,
            justification="Strong technical overlap with required stack.",
            matched_skills=["Python", "FastAPI"],
            missing_skills=["Kubernetes"],
            recommendation="Strong Match",
        )
        assert result.match_score == 8.5
        assert result.recommendation == "Strong Match"
        assert "Python" in result.matched_skills

    def test_evaluation_result_boundary_scores(self) -> None:
        """Scores at the exact boundaries (1.0 and 10.0) must be accepted."""
        low = EvaluationResult(
            match_score=1.0,
            justification="Poor fit.",
            recommendation="Not a Fit",
        )
        high = EvaluationResult(
            match_score=10.0,
            justification="Perfect fit.",
            recommendation="Strong Match",
        )
        assert low.match_score == 1.0
        assert high.match_score == 10.0

    def test_evaluation_result_invalid_score_above_max(self) -> None:
        """match_score > 10.0 must raise a ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationResult(
                match_score=10.1,          # violates le=10.0
                justification="Out of range.",
                recommendation="Strong Match",
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("match_score",) for e in errors)

    def test_evaluation_result_invalid_score_below_min(self) -> None:
        """match_score < 1.0 must raise a ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationResult(
                match_score=0.5,           # violates ge=1.0
                justification="Out of range.",
                recommendation="Not a Fit",
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("match_score",) for e in errors)

    def test_evaluation_result_invalid_recommendation(self) -> None:
        """A recommendation not in the Literal set must raise a ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationResult(
                match_score=7.0,
                justification="Okay fit.",
                recommendation="Maybe",   # not a valid Literal value
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("recommendation",) for e in errors)

    def test_empty_skill_lists_default(self) -> None:
        """matched_skills and missing_skills default to empty lists."""
        result = EvaluationResult(
            match_score=5.0,
            justification="Average.",
            recommendation="Potential Match",
        )
        assert result.matched_skills == []
        assert result.missing_skills == []


# ── CandidateRecord ───────────────────────────────────────────────────────────


class TestCandidateRecord:
    def test_candidate_record_valid(self) -> None:
        from datetime import datetime, timezone

        record = CandidateRecord(
            id="abc-123",
            job_description="Looking for a Python backend engineer.",
            parsed_resume=ParsedResume(
                full_name="Alice Chen",
                skills=["Python", "Docker"],
            ),
            evaluation=EvaluationResult(
                match_score=9.0,
                justification="Excellent match.",
                matched_skills=["Python", "Docker"],
                missing_skills=[],
                recommendation="Strong Match",
            ),
            created_at=datetime.now(timezone.utc),
        )
        assert record.id == "abc-123"
        assert record.parsed_resume.full_name == "Alice Chen"
        assert record.evaluation.match_score == 9.0

    def test_candidate_record_missing_id_raises(self) -> None:
        from datetime import datetime, timezone

        with pytest.raises(ValidationError) as exc_info:
            CandidateRecord(  # type: ignore[call-arg]
                job_description="...",
                parsed_resume=ParsedResume(full_name="Bob"),
                evaluation=EvaluationResult(
                    match_score=5.0,
                    justification=".",
                    recommendation="Potential Match",
                ),
                created_at=datetime.now(timezone.utc),
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("id",) for e in errors)
