"""
test_layout_script.py
─────────────────────
Standalone visual diagnostic script for PDF layout extraction.

Purpose
───────
Run this script directly to inspect how pdfplumber's layout=True option
represents the first page of any PDF as plain text.  Useful for verifying
that multi-column resumes are read in the correct left-to-right, top-to-bottom
order before feeding the output to the LLM.

Usage
─────
    python test_layout_script.py
    python test_layout_script.py path/to/other_resume.pdf

The file path can also be overridden via the RESUME_PATH environment variable:
    RESUME_PATH=my_resume.pdf python test_layout_script.py
"""

from __future__ import annotations

import os
import sys

import pdfplumber


# ---------------------------------------------------------------------------
# Configuration — change PDF_PATH or pass a CLI argument to override
# ---------------------------------------------------------------------------

DEFAULT_PDF_PATH = "sample_resume.pdf"


def extract_first_page_layout(pdf_path: str) -> str:
    """
    Open *pdf_path* and return the layout-aware text of its first page.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file on disk.

    Returns
    -------
    str
        Raw layout text as returned by pdfplumber.  No post-processing is
        applied so the visual spacing is exactly what the extractor service
        receives before normalisation.
    """
    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages:
            return "(PDF has no pages)"
        first_page = pdf.pages[0]
        text = first_page.extract_text(layout=True)
        return text or "(Page returned no text)"


def main() -> None:
    # Priority: CLI arg > environment variable > hardcoded default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = os.environ.get("RESUME_PATH", DEFAULT_PDF_PATH)

    if not os.path.isfile(pdf_path):
        print(f"[ERROR] File not found: {pdf_path!r}", file=sys.stderr)
        print(
            "Tip: place a PDF at the project root named 'sample_resume.pdf', "
            "or pass the path as a CLI argument.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"=== Layout extraction: {pdf_path!r} (page 1) ===\n")
    result = extract_first_page_layout(pdf_path)
    print(result)
    print("\n=== End of output ===")


if __name__ == "__main__":
    main()
