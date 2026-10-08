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
from procurement_copilot.rag.retriever import RetrievedChunk, _rerank, retrieve

from ._qdrant_fixtures import build_tmp_qdrant_index

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


# --- Unit test for the cross-encoder reranker (mocked — no real model download) ---


class _StubReranker:
    """Deterministic stand-in for sentence_transformers.CrossEncoder.predict."""

    def __init__(self, score_by_text: dict[str, float]):
        self._score_by_text = score_by_text

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self._score_by_text[text] for _, text in pairs]


class TestRerank:
    def test_reorders_by_cross_encoder_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reranking must reorder by the cross-encoder's score, not input order
        or any heading-overlap heuristic (the retired behavior)."""
        low = _make_chunk("low", "irrelevant text")
        high = _make_chunk("high", "highly relevant text")
        stub = _StubReranker({"irrelevant text": 0.1, "highly relevant text": 0.9})
        monkeypatch.setattr("procurement_copilot.rag.retriever._get_reranker", lambda: stub)

        reranked = _rerank("relevant query", [low, high])

        assert [c.chunk_id for c in reranked] == ["high", "low"]
        assert reranked[0].score == 0.9

    def test_empty_candidates_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = False

        def _should_not_be_called():
            nonlocal called
            called = True
            raise AssertionError("reranker should not load for an empty candidate list")

        monkeypatch.setattr(
            "procurement_copilot.rag.retriever._get_reranker", _should_not_be_called
        )
        assert _rerank("query", []) == []
        assert not called


class TestRerankStage2:
    def _chunks(self) -> list[RetrievedChunk]:
        # stage-1 order: a, b, c, d (scores descending)
        return [_make_chunk(i, f"text-{i}") for i in "abcd"]

    def test_rescores_only_the_shortlist_and_keeps_tail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from procurement_copilot.rag import retriever

        stub = _StubReranker({"text-a": 0.1, "text-b": 0.9, "text-c": 0.5})
        monkeypatch.setattr(retriever, "_get_reranker2", lambda: stub)
        monkeypatch.setattr(retriever.settings, "RERANK_SHORTLIST", 3)

        out = retriever._rerank_stage2("q", self._chunks())

        assert [c.chunk_id for c in out] == ["b", "c", "a", "d"]  # d never scored

    def test_noop_when_stage2_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from procurement_copilot.rag import retriever

        monkeypatch.setattr(retriever, "_get_reranker2", lambda: None)
        chunks = self._chunks()
        assert retriever._rerank_stage2("q", chunks) == chunks

    def test_falls_back_to_stage1_order_on_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from procurement_copilot.rag import retriever

        class _Boom:
            def predict(self, pairs):  # type: ignore[no-untyped-def]
                raise RuntimeError("onnx failure")

        monkeypatch.setattr(retriever, "_get_reranker2", lambda: _Boom())
        chunks = self._chunks()
        assert [c.chunk_id for c in retriever._rerank_stage2("q", chunks)] == list("abcd")

    def test_disabled_off_the_fastembed_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from procurement_copilot.rag import retriever

        monkeypatch.setattr(retriever, "_reranker2", retriever._UNSET)
        # conftest clears EMBEDDING_PROVIDER/LLM_PROVIDER, so this is not fastembed
        assert retriever._get_reranker2() is None


# --- Integration test for retrieval with a temp Qdrant index ---


@pytest.fixture()
def tmp_qdrant_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a small local Qdrant index and stub the reranker (text-length
    based, deterministic) so these tests don't download the real ~1GB
    cross-encoder model."""

    class _LengthReranker:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [float(len(text)) for _, text in pairs]

    monkeypatch.setattr(
        "procurement_copilot.rag.retriever._get_reranker", lambda: _LengthReranker()
    )

    chunks = [
        {
            "chunk_id": "far_chunk_00001",
            "text": (
                "The contracting officer must obtain approval from the agency head "
                "for sole-source contracts exceeding the simplified acquisition threshold."
            ),
            "page_start": 10,
            "page_end": 11,
            "section_heading": "6.302-1",
            "source_url": "https://www.acquisition.gov/far",
        },
        {
            "chunk_id": "far_chunk_00002",
            "text": (
                "Small business set-asides are required when the anticipated "
                "contract value exceeds $250,000 and there are at least two "
                "responsible small business concerns."
            ),
            "page_start": 42,
            "page_end": 43,
            "section_heading": "19.502-2",
            "source_url": "https://www.acquisition.gov/far",
        },
        {
            "chunk_id": "far_chunk_00003",
            "text": (
                "Cost accounting standards apply to negotiated contracts "
                "exceeding $2 million unless exempt under 48 CFR 9903."
            ),
            "page_start": 100,
            "page_end": 101,
            "section_heading": "30.201-4",
            "source_url": "https://www.acquisition.gov/far",
        },
    ]
    return build_tmp_qdrant_index(tmp_path, chunks)


class TestRetrieval:
    def test_retrieve_returns_chunks(self, tmp_qdrant_index: Path) -> None:
        results = retrieve("sole source approval", top_k=2, index_dir=tmp_qdrant_index)
        assert len(results) <= 2
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_retrieve_has_metadata(self, tmp_qdrant_index: Path) -> None:
        results = retrieve("small business", top_k=3, index_dir=tmp_qdrant_index)
        assert len(results) > 0
        first = results[0]
        assert first.chunk_id.startswith("far_chunk_")
        assert first.page_start > 0
        assert first.source_url == "https://www.acquisition.gov/far"

    def test_retrieve_returns_all_from_small_index(self, tmp_qdrant_index: Path) -> None:
        """With only 3 docs and top_k=3, all chunks should be returned."""
        results = retrieve("any query", top_k=3, index_dir=tmp_qdrant_index)
        ids = {c.chunk_id for c in results}
        assert ids == {"far_chunk_00001", "far_chunk_00002", "far_chunk_00003"}

    def test_retrieve_ranked_best_first(self, tmp_qdrant_index: Path) -> None:
        """Results must be sorted by (stub) reranker score, descending."""
        results = retrieve("any query", top_k=3, index_dir=tmp_qdrant_index)
        scores = [c.score for c in results]
        assert scores == sorted(scores, reverse=True)
