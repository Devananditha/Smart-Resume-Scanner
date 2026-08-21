"""
app/services/extractor.py
─────────────────────────
Layout-aware PDF text extraction using pdfplumber.

Design decisions
────────────────
* `layout=True` in `page.extract_text()` instructs pdfplumber to reconstruct
  the visual reading order by positioning characters according to their X/Y
  coordinates on the page.  This is essential for multi-column resumes where
  naïve extraction would interleave left- and right-column text.

* Excessive horizontal whitespace (≥ 4 consecutive spaces/tabs) is collapsed
  to a single tab character (`\\t`) so downstream parsing can use it as a
  column-separator hint while still being human-readable.

* The function is `async` to fit naturally inside FastAPI route handlers.
  pdfplumber is CPU/I-O-bound; for very large upload volumes the blocking call
  should be offloaded via `asyncio.to_thread(...)` — noted in the docstring.

* All pdfplumber and pdfminer exceptions are caught and re-raised as a FastAPI
  400 HTTPException so the caller always gets a well-formed API error.
"""

from __future__ import annotations

import io
import re

import pdfplumber
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Matches four or more consecutive horizontal-whitespace characters (spaces /
# tabs) but deliberately excludes `\\n` and `\\r` so newlines — which encode
# section breaks — are left intact.
_EXCESS_HSPACE = re.compile(r"[ \t]{4,}")


def _post_process(raw: str) -> str:
    """
    Normalise the raw layout-extracted text.

    Steps
    -----
    1. Collapse runs of ≥ 4 horizontal whitespace chars to a single tab.
       A tab is used (rather than a single space) to signal that the gap
       represents a meaningful column boundary in the original PDF.
    2. Strip trailing horizontal whitespace from every line.
    3. Strip leading / trailing blank lines from the whole document.
    """
    # Step 1 — collapse wide gaps to tab
    text = _EXCESS_HSPACE.sub("\t", raw)

    # Step 2 — remove trailing whitespace on each line (never touches newlines)
    lines = [line.rstrip() for line in text.split("\n")]

    # Step 3 — join and strip outer blank lines
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF byte string while preserving visual reading order.

    Parameters
    ----------
    file_bytes:
        Raw bytes of the uploaded PDF file.

    Returns
    -------
    str
        Post-processed plain text with layout-aware column ordering and
        collapsed horizontal whitespace.

    Raises
    ------
    HTTPException (400)
        If the bytes cannot be parsed as a valid, unencrypted PDF.

    Notes
    -----
    pdfplumber is synchronous.  In a high-throughput environment wrap this
    call with ``await asyncio.to_thread(extract_text_from_pdf_sync, ...)``
    or move extraction into a dedicated process pool.
    """
    try:
        buffer = io.BytesIO(file_bytes)
        pages_text: list[str] = []

        with pdfplumber.open(buffer) as pdf:
            for page in pdf.pages:
                # layout=True reconstructs visual reading order using
                # character X/Y coordinates — critical for multi-column PDFs.
                raw = page.extract_text(layout=True)
                if raw:
                    pages_text.append(raw)

        combined = "\n\n".join(pages_text)
        return _post_process(combined)

    except HTTPException:
        # Re-raise without wrapping (shouldn't happen here, but safety net).
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable PDF file",
        ) from exc
