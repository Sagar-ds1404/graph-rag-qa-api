"""Unit and integration tests for Graph RAG QA API using FastAPI TestClient."""

import io
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.document_processor import clean_text, chunk_document

client = TestClient(app)


def test_root_endpoint():
    """Test GET / returns 200 and expected metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Graph RAG Multi-Document Question Answering API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"


def test_health_endpoint():
    """Test GET /health returns 200 with service statuses."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "vector_store" in data
    assert "neo4j" in data
    assert "gemini" in data
    assert data["status"] == "healthy"


def test_get_documents_empty():
    """Test GET /documents returns a list (can be empty initially)."""
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_upload_invalid_file_extension():
    """Test POST /documents/upload rejects non-PDF files with 400."""
    fake_file = io.BytesIO(b"This is a text file, not a PDF.")
    response = client.post(
        "/documents/upload",
        files={"file": ("document.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 400
    assert "Only PDF documents (.pdf) are supported" in response.json()["detail"]


def test_query_validation_empty():
    """Test POST /query rejects empty or whitespace-only questions with 422."""
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 422

    response_missing = client.post("/query", json={})
    assert response_missing.status_code == 422


def test_query_gemini_unavailable():
    """Test POST /query returns 503 if Gemini is not configured."""
    with patch("app.main.is_gemini_configured", return_value=False):
        response = client.post(
            "/query",
            json={"question": "What is Graph RAG?"},
        )
        assert response.status_code == 503
        assert "Gemini LLM is not configured" in response.json()["detail"]


def test_query_successful_with_mocks():
    """Test POST /query with mocked Gemini and retrieval returns QuestionResponse."""
    mock_retrieval = {
        "vector_context": [
            {
                "text": "FastAPI is a modern web framework for Python.",
                "document": "test.pdf",
                "chunk_id": "test_c1",
                "page": 1,
            }
        ],
        "graph_context": [
            {
                "entity": "FastAPI",
                "related_entities": ["Python", "Starlette"],
            }
        ],
        "sources": [
            {
                "document": "test.pdf",
                "chunk_id": "test_c1",
                "page": 1,
            }
        ],
    }

    with patch("app.main.is_gemini_configured", return_value=True), \
         patch("app.main.retrieve_context", return_value=mock_retrieval), \
         patch("app.main.generate_grounded_answer", return_value="FastAPI is a Python web framework."):

        response = client.post(
            "/query",
            json={"question": "What is FastAPI?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is FastAPI?"
        assert data["answer"] == "FastAPI is a Python web framework."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["document"] == "test.pdf"
        assert data["sources"][0]["chunk_id"] == "test_c1"
        assert data["sources"][0]["page"] == 1


def test_graph_status_unavailable():
    """Test GET /graph/status returns 503 when Neo4j is unavailable."""
    with patch("app.main.graph_store.is_available", return_value=False):
        response = client.get("/graph/status")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unavailable"
        assert "Neo4j is not configured or reachable" in data["message"]


def test_graph_entity_lookup_unavailable():
    """Test GET /graph/entities/{name} returns 503 when Neo4j is unavailable."""
    with patch("app.main.graph_store.is_available", return_value=False):
        response = client.get("/graph/entities/FastAPI")
        assert response.status_code == 503


def test_graph_entity_lookup_success():
    """Test GET /graph/entities/{name} returns 200 when entity exists."""
    with patch("app.main.graph_store.is_available", return_value=True), \
         patch("app.main.graph_store.get_entity_and_related", return_value={
             "entity": "FastAPI",
             "related_entities": ["Python", "Uvicorn"]
         }):
        response = client.get("/graph/entities/FastAPI")
        assert response.status_code == 200
        data = response.json()
        assert data["entity"] == "FastAPI"
        assert "Python" in data["related_entities"]


def test_clean_text_utility():
    """Test text cleaning strips excess whitespace and control chars."""
    raw = "  Hello   world! \r\n\r\n This is a   test. \x00 "
    cleaned = clean_text(raw)
    assert cleaned == "Hello world!\n\nThis is a test."


def test_chunking_strategy():
    """Test chunk_document splits words correctly with page retention."""
    pages = [
        {"page": 1, "text": "Word " * 1200},
        {"page": 2, "text": "Short page text."},
    ]
    chunks = chunk_document(
        document_id="doc1",
        document_name="sample.pdf",
        pages_data=pages,
        chunk_size_words=800,
        overlap_words=100,
    )
    # Page 1 (1200 words) should yield 2 chunks; Page 2 should yield 1 chunk
    assert len(chunks) == 3
    assert chunks[0]["page"] == 1
    assert chunks[1]["page"] == 1
    assert chunks[2]["page"] == 2
    assert chunks[0]["document_id"] == "doc1"
    assert chunks[0]["chunk_id"].startswith("doc1_c")
