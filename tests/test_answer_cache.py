"""Answer cache: normalization, TTL, LRU, and run_query integration."""

import pytest

from procurement_copilot import answer_cache
from procurement_copilot.config import settings
from procurement_copilot.orchestrator import graph as graph_module
from procurement_copilot.orchestrator.graph import CopilotResponse, run_query


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "ANSWER_CACHE_TTL_SECONDS", 3600)
    monkeypatch.setattr(settings, "ANSWER_CACHE_MAX_ENTRIES", 3)
    answer_cache.clear()
    yield
    answer_cache.clear()


def _resp(q: str = "q", verified: bool = True, answer: str = "a") -> CopilotResponse:
    return CopilotResponse(question=q, intent="rag", answer=answer, is_verified=verified)


def test_normalization_ignores_case_punctuation_and_spacing() -> None:
    n = answer_cache.normalize_question
    assert n("What is the  Simplified Acquisition Threshold?") == n(
        "what is the simplified acquisition threshold"
    )
    assert n("FAR 6.302-1?") == "far 6.302-1"  # section numbers survive


def test_put_then_get_returns_an_independent_copy() -> None:
    answer_cache.put("Hello there?", _resp(answer="x"))
    got = answer_cache.get("hello there")
    assert got is not None and got.answer == "x"
    got.answer = "mutated"
    assert answer_cache.get("hello there").answer == "x"  # type: ignore[union-attr]


def test_expired_entry_is_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(answer_cache.time, "monotonic", lambda: now[0])
    answer_cache.put("q one", _resp())
    now[0] += 3601
    assert answer_cache.get("q one") is None


def test_lru_eviction_at_max_entries() -> None:
    for i in range(3):
        answer_cache.put(f"question {i}", _resp())
    answer_cache.get("question 0")  # refresh 0 so 1 is the oldest
    answer_cache.put("question 3", _resp())
    assert answer_cache.get("question 1") is None
    assert answer_cache.get("question 0") is not None


def test_ttl_zero_disables_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ANSWER_CACHE_TTL_SECONDS", 0)
    answer_cache.put("q", _resp())
    assert answer_cache.get("q") is None


class _FakeApp:
    def __init__(self, state: dict):
        self.state = state
        self.calls = 0

    def invoke(self, initial, config=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        return dict(self.state)


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, state: dict) -> _FakeApp:
    app = _FakeApp(state)
    monkeypatch.setattr(graph_module, "build_graph", lambda **kw: app)
    return app


_GOOD = {
    "intent": "rag",
    "final_answer": "Verified answer.",
    "is_verified": True,
    "rag_citations": [],
}


def test_run_query_replays_verified_answer_without_re_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _patch_pipeline(monkeypatch, _GOOD)

    first = run_query("What is the threshold?", use_cache=True)
    second = run_query("what is the threshold", use_cache=True)

    assert app.calls == 1
    assert second.answer == first.answer == "Verified answer."


def test_cache_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _patch_pipeline(monkeypatch, _GOOD)
    run_query("same question")
    run_query("same question")
    assert app.calls == 2  # default use_cache=False always runs the pipeline


@pytest.mark.parametrize(
    "state",
    [
        {**_GOOD, "is_verified": False},  # not fully verified / trimmed
        {**_GOOD, "error": "RAG error: boom"},  # transient failure
        {
            **_GOOD,
            "final_answer": "I don't have sufficiently grounded information in the FAR "
            "corpus to answer this question confidently.",
        },  # abstain
    ],
)
def test_unverified_errored_or_abstained_answers_are_never_cached(
    monkeypatch: pytest.MonkeyPatch, state: dict
) -> None:
    app = _patch_pipeline(monkeypatch, state)
    run_query("q to retry", use_cache=True)
    run_query("q to retry", use_cache=True)
    assert app.calls == 2
