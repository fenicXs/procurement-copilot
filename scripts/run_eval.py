"""RAGAS-based evaluation harness: run gold questions through the orchestrator
and score answers semantically with an LLM judge, replacing the legacy
keyword-substring matching (preserved in run_eval_legacy.py for comparison).

Usage:
    python scripts/run_eval.py [--questions eval/gold_questions.jsonl]
                               [--results-dir eval/results]
                               [--index-dir data/processed/vector_index]
                               [--db-path data/processed/procurement.duckdb]
                               [--kg-path data/processed/kg.parquet]

Metrics (via RAGAS, judged by Ollama/Gemma — EVAL_JUDGE_PROVIDER):
  - faithfulness:      are the answer's claims backed by retrieved evidence?
                        (all questions)
  - answer_relevancy:  does the answer address the question asked?
                        (all questions)
  - context_precision: were the retrieved chunks actually useful?
                        (only the subset of gold questions with a hand-written
                        `reference` answer — ~10 RAG questions)
  - context_recall:    did retrieval pull in everything needed?
                        (same reference-bearing subset)

NOTE on judge bias: the judge (Gemma) is the same model used for generation
in our live tests. This risks self-evaluation bias — a model may rate its
own style/blind-spots more favorably. Faithfulness is the metric least
exposed to this (it's closer to mechanical text-overlap-with-context
checking); answer_relevancy leans more on subjective judgment and should be
read with that caveat. See eval/results/comparison_report.md for discussion.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_QUESTIONS = _PROJECT_ROOT / "eval" / "gold_questions.jsonl"
DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "eval" / "results"
DEFAULT_INDEX_DIR = _PROJECT_ROOT / "data" / "processed" / "vector_index"
DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "processed" / "procurement.duckdb"
DEFAULT_KG_PATH = _PROJECT_ROOT / "data" / "processed" / "kg.parquet"

# This hits a single local Ollama GPU instance serving one model, not a
# scaled API — concurrency doesn't parallelize real compute here, it just
# queues requests behind each other (and risks GPU memory contention), so
# max_workers=1 is strictly sequential. Multi-step metrics like faithfulness
# chain several LLM calls per sample (extract claims, then verify each),
# and this model's per-call latency runs 30-90s, so a generous per-job
# timeout avoids spurious TimeoutErrors (confirmed empirically: 180s was too
# short and produced NaN faithfulness/context_precision/context_recall).
JUDGE_MAX_WORKERS = 1
JUDGE_TIMEOUT_SECONDS = 600


@dataclass
class EvalRow:
    """One gold question run through the orchestrator, ready for RAGAS scoring."""

    question_id: str
    category: str
    question: str
    answer: str
    contexts: list[str]
    reference: str | None
    expected_intent: str
    actual_intent: str
    intent_correct: bool
    is_verified: bool
    sql_query: str
    latency_ms: float
    error: str | None = None


def _load_questions(path: Path) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def collect_eval_rows(
    questions: list[dict],
    index_dir: Path,
    db_path: Path,
    kg_path: Path,
) -> list[EvalRow]:
    """Run each gold question through the orchestrator and collect RAGAS-ready rows."""
    from procurement_copilot.orchestrator.graph import run_query

    rows: list[EvalRow] = []

    for i, q in enumerate(questions, 1):
        qid = q.get("id", f"q_{i}")
        logger.info("[%d/%d] Running: %s", i, len(questions), qid)

        start = time.perf_counter()
        try:
            response = run_query(
                q["question"],
                index_dir=index_dir,
                db_path=db_path,
                kg_path=kg_path,
                session_id=f"ragas-eval-{qid}",
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            rows.append(
                EvalRow(
                    question_id=qid,
                    category=q.get("category", "unknown"),
                    question=q["question"],
                    answer=response.answer,
                    contexts=response.contexts or [""],  # ragas requires non-empty
                    reference=q.get("reference"),
                    expected_intent=q.get("expected_intent", ""),
                    actual_intent=response.intent,
                    intent_correct=response.intent == q.get("expected_intent", ""),
                    is_verified=response.is_verified,
                    sql_query=response.sql_query,
                    latency_ms=elapsed_ms,
                    error=response.error,
                )
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("Error evaluating %s", qid)
            rows.append(
                EvalRow(
                    question_id=qid,
                    category=q.get("category", "unknown"),
                    question=q["question"],
                    answer="",
                    contexts=[""],
                    reference=q.get("reference"),
                    expected_intent=q.get("expected_intent", ""),
                    actual_intent="error",
                    intent_correct=False,
                    is_verified=False,
                    sql_query="",
                    latency_ms=elapsed_ms,
                    error=str(exc),
                )
            )

    return rows


def build_judge():  # type: ignore[no-untyped-def]
    """Build the RAGAS-wrapped LLM judge and embeddings.

    Always Ollama today (the only free/local option we have) — the
    EVAL_JUDGE_PROVIDER setting documents the intent to force this
    regardless of what provider answers queries, keeping eval cost at zero
    even if a cloud key is configured for generation.
    """
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from procurement_copilot.config import settings

    if settings.EVAL_JUDGE_PROVIDER != "ollama":
        raise NotImplementedError(
            f"EVAL_JUDGE_PROVIDER={settings.EVAL_JUDGE_PROVIDER!r} not supported — "
            "only 'ollama' is implemented."
        )

    judge_llm = ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=0,
        num_ctx=settings.OLLAMA_NUM_CTX,
    )
    judge_embeddings = OllamaEmbeddings(
        base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_EMBED_MODEL
    )

    return (
        LangchainLLMWrapper(judge_llm),
        LangchainEmbeddingsWrapper(judge_embeddings),
    )


def _rows_to_samples(rows: list[EvalRow], require_reference: bool) -> list:  # type: ignore[no-untyped-def]
    from ragas.dataset_schema import SingleTurnSample

    samples = []
    for r in rows:
        if require_reference and not r.reference:
            continue
        samples.append(
            SingleTurnSample(
                user_input=r.question,
                response=r.answer or "(no answer produced)",
                retrieved_contexts=r.contexts,
                reference=r.reference or "",
            )
        )
    return samples


def run_ragas_metrics(samples: list, metrics: list, llm, embeddings):  # type: ignore[no-untyped-def]
    """Thin wrapper around ragas.evaluate() — isolated for testability with fakes."""
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig

    if not samples:
        return None

    dataset = EvaluationDataset(samples=samples)
    run_config = RunConfig(max_workers=JUDGE_MAX_WORKERS, timeout=JUDGE_TIMEOUT_SECONDS)

    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )


def evaluate_with_ragas(rows: list[EvalRow]) -> dict:
    """Score all rows with faithfulness/answer_relevancy, and the
    reference-bearing subset with context_precision/context_recall."""
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    llm, embeddings = build_judge()

    full_samples = _rows_to_samples(rows, require_reference=False)
    logger.info("Scoring faithfulness + answer_relevancy on %d questions...", len(full_samples))
    full_result = run_ragas_metrics(full_samples, [faithfulness, answer_relevancy], llm, embeddings)

    ref_samples = _rows_to_samples(rows, require_reference=True)
    ref_result = None
    if ref_samples:
        logger.info(
            "Scoring context_precision + context_recall on %d reference-bearing questions...",
            len(ref_samples),
        )
        ref_result = run_ragas_metrics(
            ref_samples, [context_precision, context_recall], llm, embeddings
        )
    else:
        logger.warning("No reference-bearing questions — skipping context_precision/recall.")

    return {"full": full_result, "reference_subset": ref_result}


def _mean(values: list[float]) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def generate_report(
    rows: list[EvalRow],
    ragas_results: dict,
    results_dir: Path,
) -> Path:
    """Write per-question + aggregate RAGAS scores to a timestamped JSON file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    full_df = ragas_results["full"].to_pandas() if ragas_results["full"] is not None else None
    ref_df = (
        ragas_results["reference_subset"].to_pandas()
        if ragas_results["reference_subset"] is not None
        else None
    )

    per_question = []
    for i, r in enumerate(rows):
        entry = {
            "id": r.question_id,
            "category": r.category,
            "question": r.question,
            "intent_correct": r.intent_correct,
            "is_verified": r.is_verified,
            "latency_ms": round(r.latency_ms, 1),
            "error": r.error,
        }
        if full_df is not None and i < len(full_df):
            entry["faithfulness"] = float(full_df.iloc[i]["faithfulness"])
            entry["answer_relevancy"] = float(full_df.iloc[i]["answer_relevancy"])
        per_question.append(entry)

    aggregate = {
        "total_questions": len(rows),
        "intent_accuracy": _mean([float(r.intent_correct) for r in rows]),
        "groundedness_rate": _mean([float(r.is_verified) for r in rows]),
        "avg_latency_ms": _mean([r.latency_ms for r in rows]),
    }
    if full_df is not None:
        aggregate["faithfulness"] = _mean(full_df["faithfulness"].tolist())
        aggregate["answer_relevancy"] = _mean(full_df["answer_relevancy"].tolist())
    if ref_df is not None:
        aggregate["context_precision"] = _mean(ref_df["context_precision"].tolist())
        aggregate["context_recall"] = _mean(ref_df["context_recall"].tolist())
        aggregate["reference_subset_size"] = len(ref_df)

    report = {
        "timestamp": timestamp,
        "judge": "ollama (Gemma) — same model as generation, see self-bias caveat",
        "aggregate": aggregate,
        "per_question": per_question,
    }

    output_path = results_dir / f"ragas_{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("RAGAS results written to %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation harness")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--kg-path", type=Path, default=DEFAULT_KG_PATH)
    args = parser.parse_args()

    if not args.questions.exists():
        logger.error("Questions file not found: %s", args.questions)
        sys.exit(1)

    questions = _load_questions(args.questions)
    logger.info("Loaded %d evaluation questions.", len(questions))

    rows = collect_eval_rows(questions, args.index_dir, args.db_path, args.kg_path)

    # Checkpoint raw Q&A/contexts before the slow RAGAS scoring phase — if
    # that phase gets interrupted (e.g. the allocation expires), we still
    # have the generation results instead of losing everything.
    args.results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.results_dir / "_checkpoint_rows.json"
    checkpoint_path.write_text(json.dumps([vars(r) for r in rows], indent=2), encoding="utf-8")
    logger.info("Checkpointed %d rows to %s before RAGAS scoring.", len(rows), checkpoint_path)

    ragas_results = evaluate_with_ragas(rows)
    output_path = generate_report(rows, ragas_results, args.results_dir)

    aggregate = json.loads(output_path.read_text())["aggregate"]
    print(f"\n{'='*60}")
    print("RAGAS EVALUATION SUMMARY")
    print(f"{'='*60}")
    for key, value in aggregate.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}")
    print(f"  Results: {output_path}")


if __name__ == "__main__":
    main()
    sys.exit(0)
