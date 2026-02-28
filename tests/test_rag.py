"""Tests for the RAG pipeline: chunking, retrieval, and answer generation."""

from pathlib import Path

import pytest

from procurement_copilot.rag.answer import (
    Citation,
    RAGAnswer,
    _build_citations,
    _format_context,
    generate_rag_answer,
)
from procurement_copilot.rag.retriever import RetrievedChunk, retrieve

# --- Unit tests for answer utilities ---


def _make_chunk(chunk_id: str = "far_chunk_00001", text: str = "Test text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        page_start=1,
        page_end=2,
        section_heading="Part 1",
        source_url="https://www.acquisition.gov/far",
        score=0.5,
    )


class TestFormatContext:
    def test_formats_chunks(self) -> None:
        chunks = [_make_chunk("c1", "Hello"), _make_chunk("c2", "World")]
        result = _format_context(chunks)
        assert "c1" in result
        assert "c2" in result
        assert "Hello" in result
        assert "World" in result

    def test_includes_heading(self) -> None:
        chunk = _make_chunk()
        result = _format_context([chunk])
        assert "Part 1" in result

    def test_includes_page_range(self) -> None:
        chunk = _make_chunk()
        result = _format_context([chunk])
        assert "pages 1-2" in result


class TestBuildCitations:
    def test_builds_citations(self) -> None:
        chunks = [_make_chunk("c1"), _make_chunk("c2")]
        citations = _build_citations(chunks)
        assert len(citations) == 2
        assert citations[0].chunk_id == "c1"
        assert isinstance(citations[0], Citation)


class TestGenerateRAGAnswer:
    def test_fallback_without_llm(self) -> None:
        """Without API keys, should return a structured summary."""
        from unittest.mock import patch

        chunks = [_make_chunk("c1", "Policy about approvals")]
        with patch("procurement_copilot.rag.answer._get_llm", return_value=None):
            result = generate_rag_answer("What approvals are needed?", chunks)
        assert isinstance(result, RAGAnswer)
        assert result.question == "What approvals are needed?"
        assert len(result.citations) == 1
        assert "c1" in result.answer
        assert "approvals" in result.answer.lower()


# --- Integration test for retrieval with a temp FAISS index ---


@pytest.fixture()
def tmp_faiss_index(tmp_path: Path) -> Path:
    """Build a small FAISS index from synthetic chunks for testing."""
    from langchain_community.embeddings import FakeEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="The contracting officer must obtain approval from the agency head "
            "for sole-source contracts exceeding the simplified acquisition threshold.",
            metadata={
                "chunk_id": "far_chunk_00001",
                "page_start": 10,
                "page_end": 11,
                "section_heading": "6.302-1",
                "source_url": "https://www.acquisition.gov/far",
            },
        ),
        Document(
            page_content="Small business set-asides are required when the anticipated "
            "contract value exceeds $250,000 and there are at least two responsible "
            "small business concerns.",
            metadata={
                "chunk_id": "far_chunk_00002",
                "page_start": 42,
                "page_end": 43,
                "section_heading": "19.502-2",
                "source_url": "https://www.acquisition.gov/far",
            },
        ),
        Document(
            page_content="Cost accounting standards apply to negotiated contracts "
            "exceeding $2 million unless exempt under 48 CFR 9903.",
            metadata={
                "chunk_id": "far_chunk_00003",
                "page_start": 100,
                "page_end": 101,
                "section_heading": "30.201-4",
                "source_url": "https://www.acquisition.gov/far",
            },
        ),
    ]

    embeddings = FakeEmbeddings(size=1536)
    vs = FAISS.from_documents(docs, embeddings)
    index_dir = tmp_path / "test_index"
    vs.save_local(str(index_dir))
    return index_dir


class TestRetrieval:
    def test_retrieve_returns_chunks(self, tmp_faiss_index: Path) -> None:
        results = retrieve("sole source approval", top_k=2, index_dir=tmp_faiss_index)
        assert len(results) <= 2
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_retrieve_has_metadata(self, tmp_faiss_index: Path) -> None:
        results = retrieve("small business", top_k=3, index_dir=tmp_faiss_index)
        assert len(results) > 0
        first = results[0]
        assert first.chunk_id.startswith("far_chunk_")
        assert first.page_start > 0
        assert first.source_url == "https://www.acquisition.gov/far"

    def test_retrieve_returns_all_from_small_index(self, tmp_faiss_index: Path) -> None:
        """With only 3 docs and top_k=3, all chunks should be returned."""
        results = retrieve("any query", top_k=3, index_dir=tmp_faiss_index)
        ids = {c.chunk_id for c in results}
        assert ids == {"far_chunk_00001", "far_chunk_00002", "far_chunk_00003"}
