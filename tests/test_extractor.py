"""
tests/test_extractor.py
───────────────────────
Automated tests for app/services/extractor.py.

Coverage
────────
* Happy-path: mock pdfplumber to return a controlled layout string, assert
  that post-processing (whitespace collapse) is applied correctly.
* Invalid PDF: pass raw garbage bytes and assert a 400 HTTPException is raised.
* Edge cases: empty page output, multi-page concatenation, blank-line
  preservation across pages.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from fastapi import HTTPException

from app.services.extractor import extract_text_from_pdf, _post_process


# ---------------------------------------------------------------------------
# Unit tests for _post_process (pure function — no I/O)
# ---------------------------------------------------------------------------


class TestPostProcess:
    def test_collapses_four_spaces_to_tab(self) -> None:
        assert _post_process("Name    Score") == "Name\tScore"

    def test_preserves_three_or_fewer_spaces(self) -> None:
        # Exactly 3 spaces must NOT be collapsed
        assert _post_process("A   B") == "A   B"

    def test_preserves_newlines(self) -> None:
        result = _post_process("Line1\nLine2")
        assert "\n" in result
        assert result == "Line1\nLine2"

    def test_strips_trailing_space_per_line(self) -> None:
        result = _post_process("Hello   \nWorld   ")
        # trailing spaces stripped, content kept
        assert result == "Hello\nWorld"

    def test_strips_outer_blank_lines(self) -> None:
        result = _post_process("\n\nContent\n\n")
        assert result == "Content"

    def test_wide_gap_in_multicolumn_line(self) -> None:
        # Simulates pdfplumber layout output for a two-column resume line
        raw = "Python Engineer          San Francisco, CA"
        result = _post_process(raw)
        assert "\t" in result
        assert "Python Engineer" in result
        assert "San Francisco, CA" in result


# ---------------------------------------------------------------------------
# Integration tests for extract_text_from_pdf (async)
# ---------------------------------------------------------------------------


def _make_mock_pdf(page_texts: list[str]) -> MagicMock:
    """
    Build a pdfplumber PDF mock whose pages return the given strings.

    The mock must satisfy the `with pdfplumber.open(...) as pdf:` protocol,
    so both __enter__ and __exit__ are configured.
    """
    mock_pages = []
    for text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = text
        mock_pages.append(page)

    mock_pdf = MagicMock()
    mock_pdf.pages = mock_pages
    # Context-manager protocol
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)

    return mock_pdf


class TestExtractTextFromPdf:

    @pytest.mark.asyncio
    async def test_extract_text_valid_pdf(self) -> None:
        """
        A mocked single-page PDF with a wide horizontal gap produces
        a tab-separated, whitespace-normalised string.
        """
        # Arrange — simulate pdfplumber layout output for a two-column resume
        raw_layout = "Jane Doe          jane@example.com\nPython     FastAPI     MongoDB"
        mock_pdf = _make_mock_pdf([raw_layout])

        with patch("app.services.extractor.pdfplumber.open", return_value=mock_pdf):
            result = await extract_text_from_pdf(b"%PDF-1.4 fake")

        # Assert — wide gaps collapsed to tabs, newlines preserved
        assert "Jane Doe" in result
        assert "jane@example.com" in result
        assert "\t" in result                # at least one column boundary
        assert "\n" in result                # line break preserved
        # extract_text called with layout=True on the mocked page
        mock_pdf.pages[0].extract_text.assert_called_once_with(layout=True)

    @pytest.mark.asyncio
    async def test_extract_text_multi_page_pdf(self) -> None:
        """Pages are joined by a double newline so section breaks are clear."""
        mock_pdf = _make_mock_pdf(["Page one content", "Page two content"])

        with patch("app.services.extractor.pdfplumber.open", return_value=mock_pdf):
            result = await extract_text_from_pdf(b"%PDF-1.4 fake")

        assert "Page one content" in result
        assert "Page two content" in result
        # Double newline separator between pages
        assert "\n\n" in result

    @pytest.mark.asyncio
    async def test_extract_text_skips_empty_pages(self) -> None:
        """Pages where extract_text returns None or '' are silently skipped."""
        mock_pdf = _make_mock_pdf([None, "Real content", ""])

        with patch("app.services.extractor.pdfplumber.open", return_value=mock_pdf):
            result = await extract_text_from_pdf(b"%PDF-1.4 fake")

        assert result == "Real content"

    @pytest.mark.asyncio
    async def test_extract_text_invalid_pdf(self) -> None:
        """
        Passing raw garbage bytes (not a PDF) must raise HTTPException 400.
        pdfplumber / pdfminer raises internally; the extractor re-wraps it.
        """
        with pytest.raises(HTTPException) as exc_info:
            await extract_text_from_pdf(b"not a pdf --- definitely garbage")

        assert exc_info.value.status_code == 400
        assert "Invalid or unreadable PDF" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_extract_text_pdfplumber_exception_is_wrapped(self) -> None:
        """
        Any exception from pdfplumber (e.g. encrypted PDF, corrupted stream)
        is caught and re-raised as a 400 HTTPException — never a 500.
        """
        with patch(
            "app.services.extractor.pdfplumber.open",
            side_effect=Exception("Simulated pdfplumber failure"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await extract_text_from_pdf(b"%PDF-1.4 fake")

        assert exc_info.value.status_code == 400
