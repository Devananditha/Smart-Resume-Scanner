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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.services.candidate_service import list_shortlisted_candidates, save_candidate_evaluation
from app.services.extractor import extract_text_from_pdf
from app.services.llm_service import analyze_candidate_single_pass
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


# Mount static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Liveness probe ───────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """Serve the React-less single-page frontend."""
    return FileResponse("static/index.html")


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

    Extracts raw text then runs a single-pass LLM call that simultaneously
    parses the resume into structured data AND evaluates the candidate fit.
    The result is persisted to MongoDB and returned as JSON.
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

    # 3. Single-pass LLM: extract + evaluate in one call
    analysis = await analyze_candidate_single_pass(raw_text, job_description)

    # 4. Persist to MongoDB
    record_id = await save_candidate_evaluation(
        job_description=job_description,
        parsed=analysis.parsed_resume,
        eval_result=analysis.evaluation,
    )

    return {
        "candidate_id": record_id,
        "parsed_resume": analysis.parsed_resume.model_dump(),
        "evaluation": analysis.evaluation.model_dump(),
    }


@app.get("/api/v1/candidates", tags=["Screening"])
async def get_shortlisted_candidates(min_score: float = 0.0) -> list[dict]:
    """
    Retrieve shortlisted candidates from MongoDB, sorted by descending match score.
    """
    return await list_shortlisted_candidates(min_score)
