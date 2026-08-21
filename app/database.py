"""
app/database.py
───────────────
Async MongoDB client lifecycle management using Motor.

Usage
─────
  from app.database import get_db, get_resume_collection

  # Inside a FastAPI route or service:
  db = get_db()
  collection = get_resume_collection()

Design notes
────────────
* A single `AsyncIOMotorClient` is created at import time and reused for the
  lifetime of the process — this is the recommended Motor pattern and avoids
  the overhead of reconnecting per request.
* `get_db()` and `get_resume_collection()` are thin accessors; they make it
  trivial to swap in a test database by monkey-patching `_client` in fixtures.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase

from app.config import settings

# ── Client singleton ──────────────────────────────────────────────────────────

_client: AsyncIOMotorClient = AsyncIOMotorClient(settings.mongodb_url)


# ── Accessors ─────────────────────────────────────────────────────────────────


def get_db() -> AsyncIOMotorDatabase:
    """Return the application database handle."""
    return _client[settings.database_name]


def get_resume_collection() -> AsyncIOMotorCollection:
    """Return the 'candidates' collection used to store CandidateRecord documents."""
    return get_db()["candidates"]
