"""
app/services/llm_service.py
────────────────────────────
Single-pass LLM pipeline for the Smart Resume Screener.

Architecture
────────────
Previous design: two sequential API calls (extract → evaluate).
Current design:  one API call that extracts AND evaluates simultaneously,
                 halving latency and API round-trips per resume.

Provider chain
──────────────
1. Primary   — Groq ``llama-3.3-70b-versatile`` (LPU; fast inference)
               Falls back to ``groq/compound`` if the 70B model is unavailable.
2. Secondary — Google Gemini ``gemini-3.6-flash`` (used if all Groq calls fail)

Design decisions
────────────────
* Both clients are instantiated once at module level (singletons) to reuse
  the underlying HTTP sessions across concurrent requests.

* Both ``_call_groq_single_pass`` and ``_call_gemini_single_pass`` are
  synchronous (blocking HTTP). Each is dispatched to a thread-pool executor
  via ``asyncio.to_thread`` to avoid blocking the FastAPI event loop.

* JSON output is enforced via ``response_format={"type": "json_object"}``
  for Groq (where supported) and ``response_mime_type="application/json"``
  for Gemini.  In both cases the schema is also injected into the system
  prompt so the model knows exactly which keys to emit.

* All API and JSON-parse failures are caught and re-raised as HTTP 502 so
  the router always returns a well-formed error to the client.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import HTTPException
from google import genai
from google.genai import types
from groq import Groq

from app.config import settings
from app.models.resume import CandidateAnalysis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDK clients — module-level singletons
# ---------------------------------------------------------------------------

_gemini_client = genai.Client(api_key=settings.gemini_api_key)

_groq_client: Groq | None = None
if settings.groq_api_key:
    _groq_client = Groq(api_key=settings.groq_api_key)

# Groq model preference list.
_GROQ_MODELS = [
    "llama-3.1-8b-instant",      # Preferred: LPU-accelerated, generous free-tier TPM, supports json_object
    "groq/compound",             # Groq's routing model (no json_object mode)
]

_GEMINI_MODEL = "gemini-3.6-flash"

# ---------------------------------------------------------------------------
# Combined JSON schema injected into every system prompt
# ---------------------------------------------------------------------------

_COMBINED_SCHEMA = """
{
  "parsed_resume": {
    "full_name": "string (required)",
    "email": "string or null",
    "phone": "string or null",
    "skills": ["list of skill strings"],
    "experience": [
      {
        "company": "string — required, NEVER null — omit entire entry if unknown",
        "role": "string — required, NEVER null — omit entire entry if unknown",
        "duration": "string or null",
        "highlights": ["achievement strings"]
      }
    ],
    "education": [
      {
        "institution": "string — required, NEVER null — omit entire entry if unknown",
        "degree": "string — required, NEVER null — omit entire entry if unknown",
        "graduation_year": "string or null",
        "gpa": "string or null"
      }
    ],
    "summary": "string or null"
  },
  "evaluation": {
    "match_score": <float 1.0–10.0>,
    "justification": "string — concise evidence-based reasoning",
    "matched_skills": ["skills from JD present in resume"],
    "missing_skills": ["skills from JD absent from resume"],
    "recommendation": "Strong Match" | "Potential Match" | "Not a Fit"
  }
}
"""

_SYSTEM_PROMPT = (
    "You are an expert technical recruiter and resume analyst. "
    "Given a resume and a job description, you must simultaneously:\n"
    "1. Extract all structured information from the resume.\n"
    "2. Score and evaluate the candidate's fit against the job description.\n\n"
    "Return a single JSON object that EXACTLY matches this schema "
    "(do not add extra keys, do not return null for required string fields):\n"
    f"{_COMBINED_SCHEMA}\n"
    "Rules:\n"
    "- 'recommendation' must be exactly one of: "
    "\"Strong Match\", \"Potential Match\", or \"Not a Fit\".\n"
    "- 'match_score' must be a number between 1.0 and 10.0.\n"
    "- For required sub-fields (company, role, institution, degree): "
    "if the value cannot be extracted, omit the entire parent object from its list.\n"
    "Return ONLY valid JSON — no markdown fences, no explanation, no extra keys."
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _call_groq_single_pass(user_message: str) -> str:
    """
    Attempt to run the single-pass analysis via Groq.

    Tries each model in ``_GROQ_MODELS`` in order.  Uses
    ``response_format={"type": "json_object"}`` for models that support it;
    falls back to prompt-only JSON enforcement for ``groq/compound``.

    Returns
    -------
    str
        Raw JSON string from the model.

    Raises
    ------
    RuntimeError
        If ``_groq_client`` is not configured or all models fail.
    """
    if _groq_client is None:
        raise RuntimeError("Groq client is not configured — GROQ_API_KEY is missing.")

    last_exc: Exception | None = None
    for model in _GROQ_MODELS:
        try:
            kwargs: dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                "temperature": 0.1,
            }
            # json_object mode is supported by llama models but not groq/compound
            if model != "groq/compound":
                kwargs["response_format"] = {"type": "json_object"}

            completion = _groq_client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content
        except Exception as exc:
            logger.warning("Groq model %s failed (%s). Trying next.", model, exc)
            last_exc = exc
            continue

    raise RuntimeError(f"All Groq models failed. Last error: {last_exc}")


def _call_gemini_single_pass(user_message: str) -> str:
    """
    Run the single-pass analysis via Gemini as a fallback.

    Returns
    -------
    str
        Raw JSON string from the model.
    """
    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.1,
    )
    response = _gemini_client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=user_message,
        config=config,
    )
    return response.text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def analyze_candidate_single_pass(
    resume_text: str,
    job_description: str,
) -> CandidateAnalysis:
    """
    Single-pass LLM pipeline: extract resume fields AND evaluate candidate fit
    in one API call.

    Groq is the primary provider (LPU speed).  If all Groq models fail for
    any reason, the call is transparently retried against Gemini.

    Parameters
    ----------
    resume_text:
        Plain text extracted from the candidate's PDF (layout-aware).
    job_description:
        Raw job description text provided by the recruiter.

    Returns
    -------
    CandidateAnalysis
        Validated Pydantic model containing both ``parsed_resume`` and
        ``evaluation`` fields.

    Raises
    ------
    HTTPException (502)
        If both Groq and Gemini fail, or the response cannot be parsed into
        the expected schema.
    """
    user_message = (
        f"RESUME:\n{resume_text}\n\n"
        f"JOB DESCRIPTION:\n{job_description}"
    )

    # ── Primary: Groq with Retry Logic ───────────────────────────────────────
    last_groq_exc = None
    for attempt in range(3):
        try:
            text = await asyncio.to_thread(_call_groq_single_pass, user_message)
            return CandidateAnalysis.model_validate_json(text)
        except HTTPException:
            raise
        except Exception as groq_exc:
            last_groq_exc = groq_exc
            logger.warning(
                "Groq attempt %d failed (%s: %s).",
                attempt + 1,
                type(groq_exc).__name__,
                groq_exc,
            )
            if attempt < 2:
                # Wait 4 seconds before retrying (exponential backoff for rate limits)
                await asyncio.sleep(4)

    logger.warning("All 3 Groq attempts failed. Falling back to Gemini.")

    # ── Fallback: Gemini ────────────────────────────────────────────────────
    try:
        text = await asyncio.to_thread(_call_gemini_single_pass, user_message)
        return CandidateAnalysis.model_validate_json(text)
    except Exception as gemini_exc:
        logger.error("Gemini fallback also failed: %s", gemini_exc)
        raise HTTPException(
            status_code=502,
            detail="LLM Provider Error",
        ) from gemini_exc
