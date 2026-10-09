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

    def test_partially_unsupported_answer_is_trimmed_not_discarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from procurement_copilot.orchestrator.verifier import TRIM_NOTE, verify_answer

        answer = (
            "The clause sets standard commercial contract terms and conditions. "
            "It can be tailored by the contracting officer after market research, "
            "and is incorporated by reference."
        )
        reply = (
            "GROUNDED: no\nUNSUPPORTED:\n"
            "- It can be tailored by the contracting officer after market research, "
            "and is incorporated by reference."
        )
        llm = _FakeLLM([reply])
        monkeypatch.setattr("procurement_copilot.orchestrator.verifier._get_llm", lambda: llm)

        result = verify_answer(answer, ["evidence"])
        assert not result.is_grounded  # UI still flags it as not fully verified
        assert result.verified_answer.startswith("The clause sets standard commercial")
        assert "tailored" not in result.verified_answer
        assert result.verified_answer.endswith(TRIM_NOTE)
        assert len(result.unsupported_claims) == 1

    def test_unmatched_flagged_claim_still_abstains(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from procurement_copilot.orchestrator.verifier import verify_answer

        reply = "GROUNDED: no\nUNSUPPORTED:\n- Something the answer never said"
        llm = _FakeLLM([reply])
        monkeypatch.setattr("procurement_copilot.orchestrator.verifier._get_llm", lambda: llm)

        result = verify_answer("The officer must obtain approval before award.", ["evidence"])
        assert "cannot provide a verified answer" in result.verified_answer.lower()

    def test_parse_unsupported_handles_bullets_commas_and_none(self) -> None:
        from procurement_copilot.orchestrator.verifier import _parse_unsupported

        assert _parse_unsupported("GROUNDED: yes\nUNSUPPORTED: NONE") == []
        assert _parse_unsupported("GROUNDED: no\nUNSUPPORTED:\n- one, with comma.\n- two.") == [
            "one, with comma.",
            "two.",
        ]
        assert _parse_unsupported("GROUNDED: no\nUNSUPPORTED: a claim") == ["a claim"]

    def test_unparseable_reply_is_retried_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from procurement_copilot.orchestrator.verifier import verify_answer

        llm = _FakeLLM(["", "GROUNDED: yes\nUNSUPPORTED: NONE"])
        monkeypatch.setattr("procurement_copilot.orchestrator.verifier._get_llm", lambda: llm)

        result = verify_answer("The officer must get approval.", ["evidence text"])
        assert result.is_grounded
        assert len(llm.calls) == 2

    def test_judge_exception_falls_back_to_keyword_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from procurement_copilot.orchestrator.verifier import verify_answer

        class _ErrorLLM:
            calls = 0

            def invoke(self, messages: list) -> None:
                _ErrorLLM.calls += 1
                raise RuntimeError("429 Too Many Requests")

        monkeypatch.setattr("procurement_copilot.orchestrator.verifier._get_llm", _ErrorLLM)

        # Evidence supports the answer, so the keyword fallback must verify it
        # instead of the judge failure being read as "ungrounded".
        result = verify_answer(
            "The simplified acquisition threshold is $350,000 for most procurements.",
            ["The simplified acquisition threshold is $350,000 for most procurements."],
        )
        assert result.is_grounded
        assert _ErrorLLM.calls == 2  # retried once before falling back

    def test_evidence_beyond_old_4000_char_cap_reaches_the_judge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from procurement_copilot.orchestrator.verifier import verify_answer

        llm = _FakeLLM(["GROUNDED: yes\nUNSUPPORTED: NONE"])
        monkeypatch.setattr("procurement_copilot.orchestrator.verifier._get_llm", lambda: llm)

        chunks = ["filler " * 150] * 4 + ["the late-chunk fact: $350,000"]
        verify_answer("Threshold is $350,000.", chunks)
        assert "late-chunk fact" in llm.calls[0][1].content
