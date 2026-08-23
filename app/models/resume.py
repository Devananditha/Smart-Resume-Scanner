"""
app/models/resume.py
────────────────────
Pydantic schemas for the Smart Resume Screener pipeline.

Hierarchy
─────────
  Education, WorkExperience
       └─► ParsedResume          ← structured output from Gemini
  EvaluationResult               ← scoring output from Gemini
  CandidateRecord                ← MongoDB document combining all of the above
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Sub-schemas ───────────────────────────────────────────────────────────────


class Education(BaseModel):
    """Academic background extracted from a resume."""

    institution: str = Field(
        ...,
        description="Name of the university, college, or educational institution.",
    )
    degree: str = Field(
        ...,
        description="Degree title, e.g. 'Bachelor of Science in Computer Science'.",
    )
    graduation_year: Optional[str] = Field(
        default=None,
        description="Year of graduation or expected graduation, e.g. '2023'.",
    )
    gpa: Optional[str] = Field(
        default=None,
        description="Grade Point Average as a string, e.g. '3.8/4.0'.",
    )


class WorkExperience(BaseModel):
    """A single work experience entry extracted from a resume."""

    company: str = Field(
        ...,
        description="Name of the employer or organisation.",
    )
    role: str = Field(
        ...,
        description="Job title or position held.",
    )
    duration: Optional[str] = Field(
        default=None,
        description="Employment duration, e.g. 'Jan 2021 – Mar 2023'.",
    )
    highlights: list[str] = Field(
        default_factory=list,
        description="Key responsibilities, achievements, or bullet points from the role.",
    )


# ── Core resume schema ────────────────────────────────────────────────────────


class ParsedResume(BaseModel):
    """
    Structured representation of a candidate's resume.
    This is the expected structured output from the Gemini parsing step.
    """

    full_name: str = Field(
        ...,
        description="Candidate's full name as it appears on the resume.",
    )
    email: Optional[str] = Field(
        default=None,
        description="Primary contact email address.",
    )
    phone: Optional[str] = Field(
        default=None,
        description="Primary contact phone number.",
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Flat list of technical and soft skills mentioned in the resume.",
    )
    experience: list[WorkExperience] = Field(
        default_factory=list,
        description="Ordered list of work experience entries (most recent first).",
    )
    education: list[Education] = Field(
        default_factory=list,
        description="List of educational qualifications.",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Professional summary or objective statement from the resume.",
    )


# ── Evaluation schema ─────────────────────────────────────────────────────────

RecommendationLiteral = Literal["Strong Match", "Potential Match", "Not a Fit"]


class EvaluationResult(BaseModel):
    """
    LLM-generated evaluation of how well a parsed resume fits a job description.
    """

    match_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Numeric fit score from 1.0 (poor fit) to 10.0 (perfect fit).",
    )
    justification: str = Field(
        ...,
        description="Concise explanation of the score, referencing specific evidence.",
    )
    matched_skills: list[str] = Field(
        default_factory=list,
        description="Skills from the job description that the candidate demonstrates.",
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description="Skills required by the job description that are absent from the resume.",
    )
    recommendation: RecommendationLiteral = Field(
        ...,
        description=(
            "Overall hiring recommendation: "
            "'Strong Match', 'Potential Match', or 'Not a Fit'."
        ),
    )


# ── MongoDB document schema ───────────────────────────────────────────────────


class CandidateRecord(BaseModel):
    """
    Top-level MongoDB document stored after a complete screening run.
    The `id` field maps to MongoDB's `_id` (stored as a string UUID).
    """

    id: str = Field(
        ...,
        description="Unique identifier for this candidate record (UUID string).",
    )
    job_description: str = Field(
        ...,
        description="Raw job description text used for evaluation.",
    )
    parsed_resume: ParsedResume = Field(
        ...,
        description="Structured resume data extracted by the LLM.",
    )
    evaluation: EvaluationResult = Field(
        ...,
        description="LLM-generated evaluation of the candidate against the job.",
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp of when this record was created.",
    )


# -- Single-pass LLM output ---------------------------------------------------


class CandidateAnalysis(BaseModel):
    """
    Combined output of the single-pass LLM pipeline.

    The LLM simultaneously extracts structured resume fields AND evaluates
    the candidate against the job description in one API call, returning both
    a `ParsedResume` and an `EvaluationResult` in a single JSON response.
    """

    parsed_resume: ParsedResume = Field(
        ...,
        description="Structured fields extracted from the raw resume text.",
    )
    evaluation: EvaluationResult = Field(
        ...,
        description="LLM-generated scoring and reasoning against the job description.",
    )
