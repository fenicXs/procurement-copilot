"""Offline tests for the independent-judge eval: classification, scoring, summary, payloads."""

import pytest

from scripts import eval_judge as ej


def _row(**kw) -> dict:
    base = {
        "question_id": "rag_001",
        "category": "rag",
        "question": "What is the threshold?",
        "answer": "The threshold is $350,000.",
        "contexts": ["The threshold is $350,000 for most procurements."],
        "reference": None,
        "is_verified": True,
        "intent_correct": True,
        "latency_ms": 1234.5,
        "error": None,
    }
    base.update(kw)
    return base


def _verdicts(grounded: float, answers: float = 0.95, matches: float | None = None) -> dict:
    def v(p: float) -> dict:
        return {"type": "boolean", "probabilities": {"false": 1 - p, "true": p}, "confidence": 0.9}

    out = {"grounded": v(grounded), "answers_question": v(answers)}
    if matches is not None:
        out["matches_reference"] = v(matches)
    return out


@pytest.mark.parametrize(
    "p, outcome",
    [(0.99, "supported"), (0.5, "supported"), (0.49, "unsupported"), (0.0, "unsupported")],
)
def test_classification_uses_the_judges_own_yes_no(p: float, outcome: str) -> None:
    assert ej.classify(_row(), p) == outcome


def test_abstain_error_and_missing_probability_take_precedence() -> None:
    assert ej.classify(_row(answer="I cannot provide a verified answer."), 0.99) == "abstained"
    assert (
        ej.classify(_row(answer="I don't have sufficiently grounded information in the FAR"), None)
        == "abstained"
    )
    assert ej.classify(_row(error="RAG error: boom"), 0.99) == "error"
    assert ej.classify(_row(answer="  "), 0.99) == "error"
    assert ej.classify(_row(), None) == "unsure"  # judge gave no label


def test_correct_requires_supported_and_on_topic() -> None:
    assert ej.score_row(_row(), _verdicts(0.95))["correct"] is True
    assert ej.score_row(_row(), _verdicts(0.95, answers=0.1))["correct"] is False  # evasive
    assert ej.score_row(_row(), _verdicts(0.3))["correct"] is False  # judge says not grounded


def test_high_confidence_is_a_stricter_subset_of_correct() -> None:
    borderline = ej.score_row(_row(), _verdicts(0.6))
    assert borderline["correct"] is True and borderline["correct_high_confidence"] is False
    solid = ej.score_row(_row(), _verdicts(0.9))
    assert solid["correct"] is True and solid["correct_high_confidence"] is True


def test_reference_questions_also_need_a_reference_match() -> None:
    ref_row = _row(reference="The threshold is $350,000.")
    assert ej.score_row(ref_row, _verdicts(0.95, matches=0.9))["correct"] is True
    assert ej.score_row(ref_row, _verdicts(0.95, matches=0.2))["correct"] is False


def test_unanswered_verdict_is_treated_as_unsure_not_a_crash() -> None:
    v = _verdicts(0.9)
    v["grounded"] = {
        "type": "boolean",
        "answered": False,
        "probabilities": {"false": 0.5, "true": 0.5},
    }
    assert ej.score_row(_row(), v)["outcome"] == "unsure"


def test_questions_and_state_payloads() -> None:
    assert set(ej.build_questions(False)) == {"grounded", "answers_question"}
    assert set(ej.build_questions(True)) == {"grounded", "answers_question", "matches_reference"}

    state = ej.build_state(_row(reference="ref text", contexts=["a" * 20000]))
    assert state["reference_answer"] == "ref text"
    assert len(state["evidence"]) == ej.MAX_EVIDENCE_CHARS  # capped like the production verifier
    assert "reference_answer" not in ej.build_state(_row())


def test_summary_counts_by_overall_and_category() -> None:
    scored = [
        ej.score_row(_row(question_id="a"), _verdicts(0.95)),
        ej.score_row(_row(question_id="b"), _verdicts(0.05)),
        ej.score_row(_row(question_id="c", answer="I cannot provide a verified answer."), None),
        ej.score_row(_row(question_id="d", category="sql", is_verified=False), _verdicts(0.6)),
    ]
    s = ej.summarize(scored)
    assert s["overall"]["n"] == 4
    assert s["overall"]["correct"] == 2 and s["overall"]["correct_high_confidence"] == 1
    assert s["overall"]["outcomes"] == {
        "supported": 2,
        "unsure": 0,
        "unsupported": 1,
        "abstained": 1,
        "error": 0,
    }
    assert s["overall"]["pipeline_verified"] == 3
    assert s["by_category"]["sql"]["n"] == 1 and s["by_category"]["rag"]["n"] == 3


def test_judge_rows_skips_the_api_for_abstained_and_errored_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_post(state, questions, base, key):  # type: ignore[no-untyped-def]
        calls.append(state["question"])
        return {"verdicts": _verdicts(0.95)}

    monkeypatch.setattr(ej, "post_verdicts", fake_post)
    rows = [
        _row(question_id="ok", question="q-ok"),
        _row(question_id="abst", question="q-abst", answer="I cannot provide a verified answer."),
        _row(question_id="err", question="q-err", error="boom"),
    ]
    scored = ej.judge_rows(rows, "http://x/v1", "k")

    assert calls == ["q-ok"]
    assert [s["outcome"] for s in scored] == ["supported", "abstained", "error"]


def test_sql_rows_are_judged_against_the_full_result_not_the_capped_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    ten = [(f"Agency {i}", i) for i in range(10)]
    twenty = [(f"Agency {i}", i) for i in range(20)]
    fmt = lambda rows: [f"agency: {a} | n: {n}" for a, n in rows]  # noqa: E731

    monkeypatch.setattr(
        "procurement_copilot.sql.safe_sql.execute_safe_sql",
        lambda sql, db_path=None: SimpleNamespace(error=None, columns=["agency", "n"], rows=twenty),
    )
    row = _row(
        category="sql",
        sql_query="SELECT agency, n FROM awards",
        contexts=["a FAR chunk"] + fmt(ten),  # what the pipeline kept: 10 rows
    )

    out = ej.with_full_sql_evidence(row, db_path=object())  # type: ignore[arg-type]

    assert out["contexts"][0] == "a FAR chunk"  # non-SQL evidence preserved
    assert out["contexts"][1:] == fmt(twenty)  # old 10 rows replaced by all 20, no duplicates
    assert ej.with_full_sql_evidence(row, db_path=None) is row  # no DB -> unchanged
