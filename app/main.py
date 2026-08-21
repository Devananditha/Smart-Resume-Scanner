"""
app/main.py
───────────
FastAPI application factory and entry-point.

Conventions
───────────
* Routers for individual feature areas (resume parsing, evaluation, …) are
  registered here via `app.include_router()` as they are built in later phases.
* The `/health` endpoint serves as a liveness probe for container orchestrators
  and is intentionally kept dependency-free.
"""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.candidate_service import list_shortlisted_candidates, save_candidate_evaluation
from app.services.extractor import extract_text_from_pdf
from app.services.llm_service import evaluate_candidate_fit, extract_structured_resume
from app.models.resume import CandidateRecord, EvaluationResult, ParsedResume

app = FastAPI(
    title="Smart Resume Screener",
    description="AI-powered resume parsing and candidate evaluation API.",
    version="0.1.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Liveness probe ────────────────────────────────────────────────────────────


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Returns a simple status payload used by load balancers and test suites."""
    return {"status": "healthy"}


# ── Core Endpoints ────────────────────────────────────────────────────────────


@app.post("/api/v1/screen", tags=["Screening"])
async def screen_candidate(
    file: UploadFile = File(...),
    job_description: str = Form(...),
) -> dict:
    """
    Process a candidate's resume PDF against a job description.
    
    Extracts text, parses it into structured data, evaluates the fit using an LLM,
    and stores the complete record in MongoDB.
    """
    # 1. Validate file type
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF and text files are supported.",
        )
    
    # 2. Extract text from PDF
    try:
        file_bytes = await file.read()
        raw_text = await extract_text_from_pdf(file_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {e}")

    # 3. LLM Pipeline
    # Pass 1: Parse the resume
    parsed_resume = await extract_structured_resume(raw_text)
    
    # Pass 2: Evaluate the candidate
    eval_result = await evaluate_candidate_fit(parsed_resume, job_description)
    
    # 4. Save to MongoDB
    record_id = await save_candidate_evaluation(
        job_description=job_description,
        parsed=parsed_resume,
        eval_result=eval_result
    )
    
    # Return the assembled record pieces
    return {
        "candidate_id": record_id,
        "parsed_resume": parsed_resume.model_dump(),
        "evaluation": eval_result.model_dump(),
    }


@app.get("/api/v1/candidates", tags=["Screening"])
async def get_shortlisted_candidates(min_score: float = 0.0) -> list[dict]:
    """
    Retrieve shortlisted candidates from MongoDB, sorted by descending match score.
    """
    return await list_shortlisted_candidates(min_score)
