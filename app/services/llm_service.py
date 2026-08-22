"""
app/services/llm_service.py
────────────────────────────
Two-pass LLM pipeline for the Smart Resume Screener, with an automatic Groq
fallback triggered on Gemini rate-limit (429) or server errors (502/503).

Pass 1 — Structured extraction
    Raw resume text  →  ParsedResume  (JSON mode + Pydantic validation)

Pass 2 — Semantic evaluation
    ParsedResume + job description  →  EvaluationResult  (JSON mode + Pydantic)

Design decisions
────────────────
* `genai.Client` and `groq.Groq` are both instantiated once at module level
  (singletons) so the underlying HTTP sessions are reused across requests.

* `client.models.generate_content` is synchronous (blocking HTTP).  Each call
  is offloaded to a thread-pool executor via `asyncio.to_thread` to avoid
  stalling the FastAPI event loop during network I/O.  The same applies to the
  synchronous Groq client.

* JSON output is enforced via `response_mime_type="application/json"` for Gemini
  and `response_format={"type": "json_object"}` for Groq, combined with an
  explicit JSON structure description in the system prompt.
  `response_schema=<PydanticModel>` is intentionally NOT used for Gemini: in
  google-genai SDK 2.x, passing a Pydantic model as response_schema activates
  Automatic Function Calling (AFC), which hangs or errors on
  `models.generate_content`. Instead, the response text is validated by
  `model_validate_json()` which gives identical Pydantic enforcement.

* Fallback policy:
  - If Gemini raises ANY exception (rate-limit, 503 overload, network error)
    AND a Groq API key is configured, the same prompt is replayed on Groq's
    `llama3-8b-8192` model.
  - If Groq also fails, or no Groq key is available, the original exception is
    re-raised as HTTP 502.

* All API and JSON-parse failures are caught and re-raised as HTTP 502 so
  the router always returns a well-formed error to the client.
"""

from __future__ import annotations

import asyncio
import logging
import warnings

from fastapi import HTTPException
from google import genai
from google.genai import types
from groq import Groq

from app.config import settings
from app.models.resume import EvaluationResult, ParsedResume

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDK clients — module-level singletons
# ---------------------------------------------------------------------------

_gemini_client = genai.Client(api_key=settings.gemini_api_key)

# Groq client is only functional when a key is configured.
_groq_client: Groq | None = None
if settings.groq_api_key:
    _groq_client = Groq(api_key=settings.groq_api_key)

# Primary model identifier.
_GEMINI_MODEL = "gemini-3.6-flash"

# Fallback model identifier.
# groq/compound is Groq's own production model with reliable JSON prompt following.
_GROQ_MODEL = "groq/compound"

# ---------------------------------------------------------------------------
# JSON schema hints embedded in prompts
# (used instead of response_schema= to avoid AFC activation in SDK 2.x)
# ---------------------------------------------------------------------------

_PARSED_RESUME_SCHEMA = """
{
  "full_name": "string",
  "email": "string or null",
  "phone": "string or null",
  "skills": ["string"],
  "experience": [
    {
      "company": "string (required — NEVER null. Omit this entry if company name is unknown)",
      "role": "string (required — NEVER null. Omit this entry if role is unknown)",
      "duration": "string or null",
      "highlights": ["string"]
    }
  ],
  "education": [
    {
      "institution": "string (required — NEVER null. Omit this entry if institution name is unknown)",
      "degree": "string (required — NEVER null. Omit this entry if degree is unknown)",
      "graduation_year": "string or null",
      "gpa": "string or null"
    }
  ],
  "summary": "string or null"
}
"""

_EVALUATION_RESULT_SCHEMA = """
{
  "match_score": float between 1.0 and 10.0,
  "justification": "string",
  "matched_skills": ["string"],
  "missing_skills": ["string"],
  "recommendation": "Strong Match" | "Potential Match" | "Not a Fit"
}
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _call_gemini(
    contents: types.ContentUnion,
    config: types.GenerateContentConfig,
) -> str:
    """
    Synchronous wrapper around `_gemini_client.models.generate_content`.

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
        Any network, quota, or API error from the Gemini SDK.
    """
    response = _gemini_client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=contents,
        config=config,
    )
    return response.text


def _call_groq(system_prompt: str, user_message: str) -> str:
    """
    Synchronous wrapper around the Groq ChatCompletions API.

    Instructs the Groq model to output strict JSON using the
    `json_object` response format, which is natively supported by
    llama3-8b-8192.

    Returns
    -------
    str
        The raw JSON string from the model.

    Raises
    ------
    Exception
        Any network or API error from the Groq SDK.
    """
    if _groq_client is None:
        raise RuntimeError("Groq client is not configured — GROQ_API_KEY is missing.")

    # Note: response_format={"type": "json_object"} is intentionally omitted here.
    # Not all Groq-hosted models support forced JSON validation mode.
    # JSON compliance is enforced via the system prompt instead (same strategy as Gemini).
    completion = _groq_client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract_structured_resume(raw_text: str) -> ParsedResume:
    """
    Pass 1 — convert raw resume text into a validated `ParsedResume` object.

    Attempts Gemini first; falls back to Groq (llama3-8b-8192) automatically
    on any API failure when a Groq key is configured.

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
        If both the Gemini and Groq API calls fail, or the response cannot
        be parsed into the expected schema.
    """
    system_instruction = (
        "You are an expert HR parser. "
        "Extract the following raw resume text into a JSON object that strictly "
        "matches this schema (omit missing optional fields or set them to null):\n"
        f"{_PARSED_RESUME_SCHEMA}\n"
        "Return ONLY valid JSON — no markdown, no explanation, no extra keys."
    )

    gemini_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        temperature=0.1,
    )

    # --- Primary: Gemini ---
    try:
        text = await asyncio.to_thread(_call_gemini, raw_text, gemini_config)
        return ParsedResume.model_validate_json(text)
    except HTTPException:
        raise
    except Exception as gemini_exc:
        logger.warning(
            "Gemini failed for extract_structured_resume (%s: %s). "
            "Falling back to Groq.",
            type(gemini_exc).__name__,
            gemini_exc,
        )

    # --- Fallback: Groq ---
    try:
        text = await asyncio.to_thread(_call_groq, system_instruction, raw_text)
        return ParsedResume.model_validate_json(text)
    except Exception as groq_exc:
        logger.error(
            "Groq fallback also failed for extract_structured_resume: %s", groq_exc
        )
        raise HTTPException(
            status_code=502,
            detail="LLM Provider Error",
        ) from groq_exc


async def evaluate_candidate_fit(
    parsed_resume: ParsedResume,
    job_description: str,
) -> EvaluationResult:
    """
    Pass 2 — score how well a parsed resume matches a job description.

    Attempts Gemini first; falls back to Groq (llama3-8b-8192) automatically
    on any API failure when a Groq key is configured.

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
        If both the Gemini and Groq API calls fail, or the response cannot
        be parsed into the expected schema.
    """
    system_instruction = (
        "Compare the following resume with this job description "
        "and rate fit on 1-10 with justification. "
        "Return a JSON object that strictly matches this schema:\n"
        f"{_EVALUATION_RESULT_SCHEMA}\n"
        "Return ONLY valid JSON — no markdown, no explanation, no extra keys. "
        "The 'recommendation' field MUST be exactly one of: "
        "'Strong Match', 'Potential Match', or 'Not a Fit'."
    )

    user_message = (
        f"RESUME (JSON):\n{parsed_resume.model_dump_json(indent=2)}\n\n"
        f"JOB DESCRIPTION:\n{job_description}"
    )

    gemini_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        temperature=0.2,
    )

    # --- Primary: Gemini ---
    try:
        text = await asyncio.to_thread(_call_gemini, user_message, gemini_config)
        return EvaluationResult.model_validate_json(text)
    except HTTPException:
        raise
    except Exception as gemini_exc:
        logger.warning(
            "Gemini failed for evaluate_candidate_fit (%s: %s). "
            "Falling back to Groq.",
            type(gemini_exc).__name__,
            gemini_exc,
        )

    # --- Fallback: Groq ---
    try:
        text = await asyncio.to_thread(_call_groq, system_instruction, user_message)
        return EvaluationResult.model_validate_json(text)
    except Exception as groq_exc:
        logger.error(
            "Groq fallback also failed for evaluate_candidate_fit: %s", groq_exc
        )
        raise HTTPException(
            status_code=502,
            detail="LLM Provider Error",
        ) from groq_exc
