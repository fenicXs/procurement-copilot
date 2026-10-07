"""Tests for the Corrective-RAG grade -> rewrite -> retry -> abstain loop."""

from dataclasses import dataclass

import pytest

from procurement_copilot.rag.crag import (
    GRADE_INSUFFICIENT,
    GRADE_SUFFICIENT,
    grade_documents,
    rewrite_query,
    run_crag_loop,
)
from procurement_copilot.rag.retriever import RetrievedChunk


def _make_chunk(chunk_id: str = "c1", text: str = "some FAR text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        page_start=1,
        page_end=1,
        section_heading="",
        source_url="https://www.acquisition.gov/far",
        score=1.0,
    )


@dataclass
class _FakeResponse:
    content: str


class _FakeLLM:
    """Returns canned responses in order, one per .invoke() call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages: list) -> _FakeResponse:
        self.calls.append(messages)
        return _FakeResponse(content=self._responses.pop(0))


# --- grade_documents ---


class TestGradeDocuments:
    def test_no_chunks_is_insufficient_without_calling_llm(self) -> None:
        llm = _FakeLLM([])
        assert grade_documents("q", [], llm) == GRADE_INSUFFICIENT
        assert llm.calls == []

    def test_no_llm_is_always_sufficient(self) -> None:
        assert grade_documents("q", [_make_chunk()], None) == GRADE_SUFFICIENT

    def test_llm_says_sufficient(self) -> None:
        llm = _FakeLLM(["sufficient"])
        assert grade_documents("q", [_make_chunk()], llm) == GRADE_SUFFICIENT

    def test_llm_says_insufficient(self) -> None:
        llm = _FakeLLM(["insufficient"])
        assert grade_documents("q", [_make_chunk()], llm) == GRADE_INSUFFICIENT


# --- rewrite_query ---


class TestRewriteQuery:
    def test_no_llm_returns_query_unchanged(self) -> None:
        assert rewrite_query("sole source rules", None) == "sole source rules"

    def test_llm_rewrite(self) -> None:
        llm = _FakeLLM(["FAR 6.302-1 sole source justification requirements"])
        result = rewrite_query("sole source rules", llm)
        assert result == "FAR 6.302-1 sole source justification requirements"


# --- run_crag_loop ---


class TestRunCragLoop:
    def test_sufficient_on_first_pass_no_rewrite(self) -> None:
        retrieve_calls = []

        def retrieve_fn(q: str) -> list[RetrievedChunk]:
            retrieve_calls.append(q)
            return [_make_chunk()]

        llm = _FakeLLM(["sufficient"])  # only one grade call expected
        chunks, final_query, abstained = run_crag_loop("original question", retrieve_fn, llm)

        assert not abstained
        assert len(chunks) == 1
        assert retrieve_calls == ["original question"]
        assert final_query == "original question"
        assert len(llm.calls) == 1  # grade only, no rewrite

    def test_insufficient_n_times_then_abstain(self) -> None:
        retrieve_calls = []

        def retrieve_fn(q: str) -> list[RetrievedChunk]:
            retrieve_calls.append(q)
            return [_make_chunk()]

        max_retries = 2
        # grade, rewrite, grade, rewrite, grade (always insufficient / always rewritten)
        llm = _FakeLLM(
            [
                "insufficient",
                "rewritten query 1",
                "insufficient",
                "rewritten query 2",
                "insufficient",
            ]
        )

        chunks, final_query, abstained = run_crag_loop(
            "original question", retrieve_fn, llm, max_retries=max_retries
        )

        assert abstained
        assert len(retrieve_calls) == max_retries + 1
        assert final_query == "rewritten query 2"
        # exactly max_retries rewrite calls happened (interleaved with grades)
        assert retrieve_calls == ["original question", "rewritten query 1", "rewritten query 2"]

    def test_recovers_after_one_rewrite(self) -> None:
        def retrieve_fn(q: str) -> list[RetrievedChunk]:
            return [_make_chunk(text=q)]

        llm = _FakeLLM(["insufficient", "better query", "sufficient"])
        chunks, final_query, abstained = run_crag_loop("vague question", retrieve_fn, llm)

        assert not abstained
        assert final_query == "better query"
        assert chunks[0].text == "better query"


# --- LLM-based groundedness verifier ---


class TestVerifyAnswerLLM:
    def test_llm_grounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from procurement_copilot.orchestrator.verifier import verify_answer

        llm = _FakeLLM(["GROUNDED: yes\nUNSUPPORTED: NONE"])
        monkeypatch.setattr("procurement_copilot.orchestrator.verifier._get_llm", lambda: llm)

        result = verify_answer("The officer must get approval.", ["evidence text"])
        assert result.is_grounded
        assert result.unsupported_claims == []
        assert result.verified_answer == "The officer must get approval."

    def test_llm_ungrounded_abstains(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from procurement_copilot.orchestrator.verifier import verify_answer

        llm = _FakeLLM(["GROUNDED: no\nUNSUPPORTED: Quantum computing is used here"])
        monkeypatch.setattr("procurement_copilot.orchestrator.verifier._get_llm", lambda: llm)

        result = verify_answer("Quantum computing is used here.", ["unrelated evidence"])
        assert not result.is_grounded
        assert "cannot provide a verified answer" in result.verified_answer.lower()
        assert result.unsupported_claims == ["Quantum computing is used here"]
