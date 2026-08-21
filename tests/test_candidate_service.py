"""
tests/test_candidate_service.py
───────────────────────────────
Unit tests for the MongoDB candidate service.
"""

from unittest.mock import AsyncMock, patch

import pytest
from app.models.resume import EvaluationResult, ParsedResume
from app.services.candidate_service import list_shortlisted_candidates, save_candidate_evaluation


@pytest.fixture
def mock_collection():
    with patch("app.services.candidate_service.get_resume_collection") as mock_get_col:
        mock_col = AsyncMock()
        mock_get_col.return_value = mock_col
        yield mock_col


@pytest.fixture
def sample_parsed_resume():
    return ParsedResume(
        full_name="John Doe",
        email="john@example.com",
        phone=None,
        skills=["Python", "FastAPI"],
        experience=[],
        education=[],
        summary=None,
    )


@pytest.fixture
def sample_evaluation():
    return EvaluationResult(
        match_score=8.5,
        justification="Strong technical match.",
        matched_skills=["Python"],
        missing_skills=["MongoDB"],
        recommendation="Strong Match",
    )


@pytest.mark.asyncio
async def test_save_candidate_evaluation(mock_collection, sample_parsed_resume, sample_evaluation):
    jd = "Backend Developer"
    
    # Run the service function
    record_id = await save_candidate_evaluation(jd, sample_parsed_resume, sample_evaluation)
    
    # Ensure it returns a string ID
    assert isinstance(record_id, str)
    assert len(record_id) > 0
    
    # Ensure insert_one was called once
    mock_collection.insert_one.assert_called_once()
    
    # Extract the inserted document
    inserted_doc = mock_collection.insert_one.call_args[0][0]
    
    # Verify the document structure
    assert inserted_doc["_id"] == record_id
    assert "id" not in inserted_doc
    assert inserted_doc["job_description"] == jd
    assert inserted_doc["parsed_resume"]["full_name"] == "John Doe"
    assert inserted_doc["evaluation"]["match_score"] == 8.5
    assert "created_at" in inserted_doc


@pytest.mark.asyncio
async def test_list_shortlisted_candidates(mock_collection):
    from unittest.mock import MagicMock
    
    # Setup mock cursor (find is synchronous in Motor, it returns a cursor)
    mock_cursor = MagicMock()
    mock_collection.find = MagicMock(return_value=mock_cursor)
    mock_cursor.sort.return_value = mock_cursor
    
    # Setup mock async iteration
    mock_cursor.__aiter__.return_value = [
        {"_id": "123", "evaluation": {"match_score": 9.0}},
        {"_id": "456", "evaluation": {"match_score": 7.5}}
    ]
    
    results = await list_shortlisted_candidates(min_score=7.0)
    
    # Verify find was called with correct query
    mock_collection.find.assert_called_once_with({"evaluation.match_score": {"$gte": 7.0}})
    
    # Verify sort was called
    mock_cursor.sort.assert_called_once_with("evaluation.match_score", -1)
    
    # Verify results
    assert len(results) == 2
    assert results[0]["id"] == "123"
    assert "_id" not in results[0]
    assert results[1]["id"] == "456"
