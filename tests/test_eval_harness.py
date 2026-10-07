"""Smoke tests for the RAGAS eval harness's own wiring (sample building, judge
construction, report aggregation) — offline, no live Ollama or network calls.

NOTE: RAGAS's actual metrics (faithfulness, answer_relevancy, ...) do
multi-step structured LLM calls with an internal prompt/parsing contract
specific to the installed ragas version. Faking that convincingly would
mostly test ragas's own internals, not ours — so these tests fake the
*shape* of a ragas EvaluationResult (anything with `.to_pandas()`) rather
than exercising the real metric computation.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from procurement_copilot.config import settings
from scripts.run_eval import EvalRow, _rows_to_samples, build_judge, generate_report


def _make_row(qid: str, reference: str | None = None, verified: bool = True) -> EvalRow:
    return EvalRow(
        question_id=qid,
        category="rag",
        question=f"Question for {qid}?",
        answer=f"Answer for {qid}.",
        contexts=[f"Context for {qid}."],
        reference=reference,
        expected_intent="rag",
        actual_intent="rag",
        intent_correct=True,
        is_verified=verified,
        sql_query="",
        latency_ms=100.0,
    )


class _FakeRagasResult:
    """Stands in for ragas's EvaluationResult — only `.to_pandas()` is used."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df


class TestRowsToSamples:
    def test_includes_all_rows_without_reference_requirement(self) -> None:
        rows = [_make_row("q1"), _make_row("q2", reference="some reference")]
        samples = _rows_to_samples(rows, require_reference=False)
        assert len(samples) == 2

    def test_filters_to_reference_bearing_subset(self) -> None:
        rows = [_make_row("q1"), _make_row("q2", reference="some reference")]
        samples = _rows_to_samples(rows, require_reference=True)
        assert len(samples) == 1
        assert samples[0].user_input == "Question for q2?"

    def test_empty_reference_subset(self) -> None:
        rows = [_make_row("q1"), _make_row("q2")]
        samples = _rows_to_samples(rows, require_reference=True)
        assert samples == []


class TestBuildJudge:
    def test_raises_for_unsupported_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "EVAL_JUDGE_PROVIDER", "openai")
        with pytest.raises(NotImplementedError):
            build_judge()

    def test_builds_ollama_judge_by_default(self) -> None:
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        assert settings.EVAL_JUDGE_PROVIDER == "ollama"
        llm_wrapper, embeddings_wrapper = build_judge()

        assert isinstance(llm_wrapper.langchain_llm, ChatOllama)
        assert isinstance(embeddings_wrapper.embeddings, OllamaEmbeddings)


class TestGenerateReport:
    def test_aggregates_full_and_reference_subset(self, tmp_path: Path) -> None:
        rows = [
            _make_row("q1", verified=True),
            _make_row("q2", reference="ref answer", verified=False),
        ]
        full_df = pd.DataFrame({"faithfulness": [0.8, 0.6], "answer_relevancy": [0.9, 0.7]})
        ref_df = pd.DataFrame({"context_precision": [0.75], "context_recall": [0.5]})
        ragas_results = {
            "full": _FakeRagasResult(full_df),
            "reference_subset": _FakeRagasResult(ref_df),
        }

        output_path = generate_report(rows, ragas_results, tmp_path)
        report = json.loads(output_path.read_text())

        assert report["aggregate"]["total_questions"] == 2
        assert report["aggregate"]["groundedness_rate"] == 0.5
        assert report["aggregate"]["faithfulness"] == pytest.approx(0.7)
        assert report["aggregate"]["answer_relevancy"] == pytest.approx(0.8)
        assert report["aggregate"]["context_precision"] == pytest.approx(0.75)
        assert report["aggregate"]["context_recall"] == pytest.approx(0.5)
        assert report["aggregate"]["reference_subset_size"] == 1
        assert len(report["per_question"]) == 2
        assert report["per_question"][0]["faithfulness"] == pytest.approx(0.8)

    def test_handles_missing_reference_subset(self, tmp_path: Path) -> None:
        rows = [_make_row("q1")]
        full_df = pd.DataFrame({"faithfulness": [0.9], "answer_relevancy": [0.85]})
        ragas_results = {"full": _FakeRagasResult(full_df), "reference_subset": None}

        output_path = generate_report(rows, ragas_results, tmp_path)
        report = json.loads(output_path.read_text())

        assert "context_precision" not in report["aggregate"]
        assert "context_recall" not in report["aggregate"]
        assert report["aggregate"]["faithfulness"] == pytest.approx(0.9)
