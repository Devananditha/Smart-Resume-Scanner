"""
app/services/llm_service.py
────────────────────────────
Two-pass Gemini LLM pipeline for the Smart Resume Screener.

Pass 1 — Structured extraction
    Raw resume text  →  ParsedResume  (schema-constrained JSON output)

Pass 2 — Semantic evaluation
    ParsedResume + job description  →  EvaluationResult  (schema-constrained)

Design decisions
────────────────
* `genai.Client` is instantiated once at module level (singleton) so the
  underlying HTTP session is reused across requests.

* `client.models.generate_content` is synchronous (blocking HTTP).  Each call
  is offloaded to a thread-pool executor via `asyncio.to_thread` to avoid
  stalling the FastAPI event loop during network I/O.

* Schema-constrained output (`response_mime_type="application/json"` +
  `response_schema=<PydanticModel>`) instructs Gemini to emit JSON that
  conforms to the Pydantic model's JSON Schema.  The SDK handles schema
  serialisation internally.

* All API and JSON-parse failures are caught and re-raised as HTTP 502 so
  the router always returns a well-formed error to the client.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import HTTPException
from google import genai
from google.genai import types

from app.config import settings
from app.models.resume import EvaluationResult, ParsedResume

# ---------------------------------------------------------------------------
# SDK client — module-level singleton
# ---------------------------------------------------------------------------

client = genai.Client(api_key=settings.gemini_api_key)

# Model identifier — change here to switch Gemini versions project-wide.
_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _call_generate(
    contents: types.ContentUnion,
    config: types.GenerateContentConfig,
) -> str:
    """
    Synchronous wrapper around `client.models.generate_content`.

    Extracted as a named function (rather than a lambda) so that
    `asyncio.to_thread` can reference it cleanly and test mocks can
    patch it at a single call-site.

    Returns
    -------
    str
        The raw text of the first response candidate.

    Raises
    ------
    Exception
        Any network, quota, or API error from the Gemini SDK — callers are
        responsible for wrapping these in an appropriate HTTPException.
    """
    response = client.models.generate_content(
        model=_MODEL,
        contents=contents,
        config=config,
    )
    return response.text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract_structured_resume(raw_text: str) -> ParsedResume:
    """
    Pass 1 — convert raw resume text into a validated `ParsedResume` object.

    Parameters
    ----------
    raw_text:
        Plain text extracted from the candidate's PDF (layout-aware).

    Returns
    -------
    ParsedResume
        Validated Pydantic model populated from the LLM's JSON output.

    Raises
    ------
    HTTPException (502)
        If the Gemini API call fails or the response cannot be parsed.
    """
    system_instruction = (
        "You are an expert HR parser. "
        "Extract the following raw resume text into the requested JSON schema. "
        "Be highly accurate. If a field is missing, leave it empty."
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=ParsedResume,
        temperature=0.1,   # low temperature for deterministic extraction
    )

    try:
        text = await asyncio.to_thread(_call_generate, raw_text, config)
        return ParsedResume.model_validate_json(text)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="LLM Provider Error",
        ) from exc


async def evaluate_candidate_fit(
    parsed_resume: ParsedResume,
    job_description: str,
) -> EvaluationResult:
    """
    Pass 2 — score how well a parsed resume matches a job description.

    Parameters
    ----------
    parsed_resume:
        Structured resume data produced by `extract_structured_resume`.
    job_description:
        Raw job description text provided by the recruiter.

    Returns
    -------
    EvaluationResult
        Validated Pydantic model containing the match score, justification,
        matched/missing skills, and hiring recommendation.

    Raises
    ------
    HTTPException (502)
        If the Gemini API call fails or the response cannot be parsed.
    """
    system_instruction = (
        "Compare the following resume with this job description "
        "and rate fit on 1-10 with justification."
    )

    # Build the user message from both inputs so the model sees both in context.
    user_message = (
        f"RESUME (JSON):\n{parsed_resume.model_dump_json(indent=2)}\n\n"
        f"JOB DESCRIPTION:\n{job_description}"
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=EvaluationResult,
        temperature=0.2,
    )

    try:
        text = await asyncio.to_thread(_call_generate, user_message, config)
        return EvaluationResult.model_validate_json(text)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="LLM Provider Error",
        ) from exc
