"""
app/services/llm_service.py
────────────────────────────
Single-pass LLM pipeline for the Smart Resume Screener.

Architecture
────────────
One API call that extracts AND evaluates simultaneously, halving latency and API round-trips.

Provider chain
──────────────
1. Primary   — OpenRouter `meta-llama/llama-3.1-8b-instruct:free`
2. Secondary — Google Gemini `gemini-1.5-flash-latest` (used if OpenRouter fails)
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import HTTPException
from google import genai
from google.genai import types
from openai import AsyncOpenAI

from app.config import settings
from app.models.resume import CandidateAnalysis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDK clients — module-level singletons
# ---------------------------------------------------------------------------

_gemini_client = genai.Client(api_key=settings.gemini_api_key)

openrouter_client = AsyncOpenAI(
    api_key=settings.openrouter_api_key or "DUMMY",
    base_url="https://openrouter.ai/api/v1"
)

_GEMINI_MODEL = "gemini-1.5-flash"

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


def _call_gemini_single_pass(user_message: str) -> str:
    """
    Run the single-pass analysis via Gemini as a fallback.
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
    Single-pass LLM pipeline: extract resume fields AND evaluate candidate fit.

    OpenRouter is primary. Gemini is secondary.
    """
    user_message = (
        f"RESUME:\n{resume_text}\n\n"
        f"JOB DESCRIPTION:\n{job_description}"
    )

    # ── Primary Attempt (OpenRouter) ─────────────────────────────────────────
    try:
        response = await openrouter_client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct:free",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        text = response.choices[0].message.content
        return CandidateAnalysis.model_validate_json(text)
    except Exception as or_exc:
        logger.warning("OpenRouter attempt failed: %s", or_exc)
        await asyncio.sleep(2)

    # ── Fallback Attempt (Gemini) ────────────────────────────────────────────
    try:
        text = await asyncio.to_thread(_call_gemini_single_pass, user_message)
        return CandidateAnalysis.model_validate_json(text)
    except Exception as gemini_exc:
        logger.error("Gemini fallback also failed: %s", gemini_exc)
        raise HTTPException(
            status_code=502,
            detail="LLM Provider Error",
        ) from gemini_exc
