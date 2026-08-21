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

from fastapi import FastAPI

app = FastAPI(
    title="Smart Resume Screener",
    description="AI-powered resume parsing and candidate evaluation API.",
    version="0.1.0",
)


# ── Liveness probe ────────────────────────────────────────────────────────────


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Returns a simple status payload used by load balancers and test suites."""
    return {"status": "healthy"}
