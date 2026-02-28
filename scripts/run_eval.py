"""Evaluation harness: run gold questions through the orchestrator and report metrics.

Usage:
    python scripts/run_eval.py [--questions eval/gold_questions.jsonl]
                               [--report eval/report.md]
                               [--index-dir data/processed/vector_index]
                               [--db-path data/processed/procurement.duckdb]
                               [--kg-path data/processed/kg.parquet]

Produces eval/report.md with retrieval, groundedness, SQL success, and
intent-routing metrics.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_QUESTIONS = _PROJECT_ROOT / "eval" / "gold_questions.jsonl"
DEFAULT_REPORT = _PROJECT_ROOT / "eval" / "report.md"
DEFAULT_INDEX_DIR = _PROJECT_ROOT / "data" / "processed" / "vector_index"
DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "processed" / "procurement.duckdb"
DEFAULT_KG_PATH = _PROJECT_ROOT / "data" / "processed" / "kg.parquet"


@dataclass
class QuestionResult:
    """Result of evaluating a single question."""

    question_id: str
    category: str
    question: str
    expected_intent: str
    actual_intent: str
    intent_correct: bool
    answer: str
    has_answer: bool
    keyword_hits: int
    keyword_total: int
    keyword_recall: float
    is_verified: bool
    sql_query: str
    sql_success: bool
    latency_ms: float
    error: str | None = None


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""

    total_questions: int = 0
    intent_accuracy: float = 0.0
    answer_rate: float = 0.0
    avg_keyword_recall: float = 0.0
    groundedness_rate: float = 0.0
    sql_success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    results_by_category: dict[str, dict] = field(default_factory=dict)


def _load_questions(path: Path) -> list[dict]:
    """Load gold questions from JSONL."""
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def _compute_keyword_recall(answer: str, expected_keywords: list[str]) -> tuple[int, int, float]:
    """Compute how many expected keywords appear in the answer."""
    if not expected_keywords:
        return 0, 0, 1.0

    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    total = len(expected_keywords)
    recall = hits / total if total > 0 else 0.0
    return hits, total, recall


def evaluate_questions(
    questions: list[dict],
    index_dir: Path,
    db_path: Path,
    kg_path: Path,
) -> list[QuestionResult]:
    """Run each question through the orchestrator and collect results."""
    from procurement_copilot.orchestrator.graph import run_query

    results: list[QuestionResult] = []

    for i, q in enumerate(questions, 1):
        qid = q.get("id", f"q_{i}")
        category = q.get("category", "unknown")
        question = q["question"]
        expected_intent = q.get("expected_intent", "")
        expected_keywords = q.get("expected_keywords", [])

        logger.info("[%d/%d] Evaluating: %s", i, len(questions), qid)

        start = time.perf_counter()
        try:
            response = run_query(
                question,
                index_dir=index_dir,
                db_path=db_path,
                kg_path=kg_path,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            hits, total, recall = _compute_keyword_recall(response.answer, expected_keywords)

            sql_success = True
            if category == "sql" and response.sql_query:
                sql_success = "error" not in response.answer.lower()

            results.append(
                QuestionResult(
                    question_id=qid,
                    category=category,
                    question=question,
                    expected_intent=expected_intent,
                    actual_intent=response.intent,
                    intent_correct=response.intent == expected_intent,
                    answer=response.answer,
                    has_answer=len(response.answer) > 0,
                    keyword_hits=hits,
                    keyword_total=total,
                    keyword_recall=recall,
                    is_verified=response.is_verified,
                    sql_query=response.sql_query,
                    sql_success=sql_success,
                    latency_ms=elapsed_ms,
                    error=response.error,
                )
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("Error evaluating %s", qid)
            results.append(
                QuestionResult(
                    question_id=qid,
                    category=category,
                    question=question,
                    expected_intent=expected_intent,
                    actual_intent="error",
                    intent_correct=False,
                    answer="",
                    has_answer=False,
                    keyword_hits=0,
                    keyword_total=len(expected_keywords),
                    keyword_recall=0.0,
                    is_verified=False,
                    sql_query="",
                    sql_success=False,
                    latency_ms=elapsed_ms,
                    error=str(exc),
                )
            )

    return results


def compute_metrics(results: list[QuestionResult]) -> EvalMetrics:
    """Aggregate individual results into summary metrics."""
    n = len(results)
    if n == 0:
        return EvalMetrics()

    metrics = EvalMetrics(total_questions=n)
    metrics.intent_accuracy = sum(r.intent_correct for r in results) / n
    metrics.answer_rate = sum(r.has_answer for r in results) / n
    metrics.avg_keyword_recall = sum(r.keyword_recall for r in results) / n
    metrics.groundedness_rate = sum(r.is_verified for r in results) / n
    metrics.avg_latency_ms = sum(r.latency_ms for r in results) / n

    # SQL success rate (only for SQL questions)
    sql_results = [r for r in results if r.category == "sql"]
    if sql_results:
        metrics.sql_success_rate = sum(r.sql_success for r in sql_results) / len(sql_results)

    # Per-category breakdown
    categories = sorted({r.category for r in results})
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cat_n = len(cat_results)
        metrics.results_by_category[cat] = {
            "count": cat_n,
            "intent_accuracy": sum(r.intent_correct for r in cat_results) / cat_n,
            "answer_rate": sum(r.has_answer for r in cat_results) / cat_n,
            "avg_keyword_recall": sum(r.keyword_recall for r in cat_results) / cat_n,
            "groundedness_rate": sum(r.is_verified for r in cat_results) / cat_n,
            "avg_latency_ms": sum(r.latency_ms for r in cat_results) / cat_n,
        }

    return metrics


def generate_report(metrics: EvalMetrics, results: list[QuestionResult], output_path: Path) -> None:
    """Write evaluation report as Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Procurement Copilot — Evaluation Report\n")
    lines.append(f"**Total questions evaluated:** {metrics.total_questions}\n")

    # Overall metrics table
    lines.append("## Overall Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Intent Routing Accuracy | {metrics.intent_accuracy:.1%} |")
    lines.append(f"| Answer Rate | {metrics.answer_rate:.1%} |")
    lines.append(f"| Avg Keyword Recall | {metrics.avg_keyword_recall:.1%} |")
    lines.append(f"| Groundedness Rate | {metrics.groundedness_rate:.1%} |")
    lines.append(f"| SQL Execution Success | {metrics.sql_success_rate:.1%} |")
    lines.append(f"| Avg Latency | {metrics.avg_latency_ms:.0f} ms |")
    lines.append("")

    # Per-category breakdown
    lines.append("## Results by Category\n")
    lines.append(
        "| Category | Count | Intent Acc | Answer Rate "
        "| Keyword Recall | Grounded | Avg Latency |"
    )
    lines.append(
        "|----------|-------|------------|-------------|"
        "----------------|----------|-------------|"
    )
    for cat, stats in sorted(metrics.results_by_category.items()):
        lines.append(
            f"| {cat} | {stats['count']} "
            f"| {stats['intent_accuracy']:.1%} "
            f"| {stats['answer_rate']:.1%} "
            f"| {stats['avg_keyword_recall']:.1%} "
            f"| {stats['groundedness_rate']:.1%} "
            f"| {stats['avg_latency_ms']:.0f} ms |"
        )
    lines.append("")

    # Sample results (first 3 per category)
    lines.append("## Sample Results\n")
    categories = sorted({r.category for r in results})
    for cat in categories:
        lines.append(f"### {cat.upper()} Examples\n")
        cat_results = [r for r in results if r.category == cat][:3]
        for r in cat_results:
            lines.append(f"**Q ({r.question_id}):** {r.question}\n")
            lines.append(
                f"- **Intent:** {r.actual_intent} "
                f"({'correct' if r.intent_correct else 'WRONG'})"
            )
            lines.append(
                f"- **Keyword Recall:** {r.keyword_recall:.0%} "
                f"({r.keyword_hits}/{r.keyword_total})"
            )
            lines.append(f"- **Verified:** {'Yes' if r.is_verified else 'No'}")
            lines.append(f"- **Latency:** {r.latency_ms:.0f} ms")
            if r.sql_query:
                lines.append(f"- **SQL:** `{r.sql_query}`")
            if r.error:
                lines.append(f"- **Error:** {r.error}")
            # Truncated answer preview
            preview = r.answer[:200].replace("\n", " ")
            if len(r.answer) > 200:
                preview += "..."
            lines.append(f"- **Answer preview:** {preview}")
            lines.append("")

    # Errors summary
    errors = [r for r in results if r.error]
    if errors:
        lines.append("## Errors\n")
        for r in errors:
            lines.append(f"- **{r.question_id}**: {r.error}")
        lines.append("")

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
    logger.info("Report written to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--kg-path", type=Path, default=DEFAULT_KG_PATH)
    args = parser.parse_args()

    if not args.questions.exists():
        logger.error("Questions file not found: %s", args.questions)
        sys.exit(1)

    questions = _load_questions(args.questions)
    logger.info("Loaded %d evaluation questions.", len(questions))

    results = evaluate_questions(
        questions,
        index_dir=args.index_dir,
        db_path=args.db_path,
        kg_path=args.kg_path,
    )

    metrics = compute_metrics(results)
    generate_report(metrics, results, args.report)

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Questions:          {metrics.total_questions}")
    print(f"  Intent Accuracy:    {metrics.intent_accuracy:.1%}")
    print(f"  Answer Rate:        {metrics.answer_rate:.1%}")
    print(f"  Keyword Recall:     {metrics.avg_keyword_recall:.1%}")
    print(f"  Groundedness:       {metrics.groundedness_rate:.1%}")
    print(f"  SQL Success:        {metrics.sql_success_rate:.1%}")
    print(f"  Avg Latency:        {metrics.avg_latency_ms:.0f} ms")
    print(f"{'='*60}")
    print(f"  Report: {args.report}")


if __name__ == "__main__":
    main()
    sys.exit(0)
