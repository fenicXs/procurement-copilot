"""Independent-judge evaluation: run every gold question through the current
pipeline (as deployed: Groq primary, Gemini fallback, no answer cache) and judge
each answer with ASU Research Computing's Verdicts API.

Why Verdicts: it returns a calibrated true/false *probability* from a model
trained for decisions (read from the first token, so there is no reply to parse
and no reasoning-model quirks), and it is a different model family from the
generator — unlike the RAGAS run, which used Gemma to judge Gemma.

Per question the judge answers independent boolean questions:
  - grounded:        is every specific fact in the answer supported by the
                     retrieved evidence?                       (all questions)
  - answers_question: does the answer actually address the question? (all)
  - matches_reference: does it agree with the hand-written reference? (the
                     ~10 questions that have one)

Outcome per question:
  abstained   the system declined to answer
  error       the pipeline raised / returned an error
  supported   the judge says "grounded" (p >= SUPPORTED_P = 0.5, the judge's own yes)
  unsupported the judge says "not grounded" (p < 0.5)
  unsure      the judge gave no usable label
A question is counted CORRECT when it is `supported` AND the judge agrees it
answers the question AND (if a reference exists) it matches the reference.
`correct_high_confidence` additionally requires p(grounded) >= CONFIDENT_P (0.8);
the gap between the two counts is the borderline band, reported rather than hidden.

SQL answers are judged against the full result rows the narration saw (the
pipeline's own evidence list is capped at 10 rows, which would make the judge
flag correct statements about rows 11-20).

Usage (from procurement-copilot/):
    python scripts/eval_judge.py [--limit N] [--results-dir eval/results] [--regenerate]

Generation is checkpointed per question to eval/results/_judge_rows.json, so a
Groq daily-token cap or a crash does not lose work: re-run to resume.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_judge")

_DEFAULT_INDEX = _PROJECT_ROOT / "data" / "processed" / "vector_index_cloud"
SUPPORTED_P = 0.5  # the judge's own boolean verdict (value is true when p > 0.5)
CONFIDENT_P = 0.8
MATCH_P = 0.5
SQL_EVIDENCE_ROWS = 20  # what the SQL narration step sees
MAX_EVIDENCE_CHARS = 12000
JUDGE_WORKERS = 4

ABSTAIN_PREFIXES = (
    "I don't have sufficiently grounded information",
    "I cannot provide a verified answer",
    "I could not find relevant information",
    "No relevant knowledge graph triples found",
)


# --------------------------------------------------------------------------
# Verdicts API
# --------------------------------------------------------------------------


def build_questions(has_reference: bool) -> dict:
    qs = {
        "grounded": {
            "type": "boolean",
            "instructions": (
                "Is every specific fact in the answer (numbers, thresholds, authorities, "
                "citations, requirements, names) supported by the evidence?"
            ),
            "criteria": {
                "true": "Every specific fact in the answer appears in, or is clearly implied by, "
                "the evidence",
                "false": "The answer states a specific number, requirement, authority or fact that "
                "is not in the evidence or contradicts it",
            },
        },
        "answers_question": {
            "type": "boolean",
            "instructions": "Does the answer directly address the question that was asked?",
            "criteria": {
                "true": "The answer gives the information the question asks for",
                "false": "The answer is off-topic, evasive, or says it cannot answer",
            },
        },
    }
    if has_reference:
        qs["matches_reference"] = {
            "type": "boolean",
            "instructions": "Does the answer agree with the reference answer on the key facts?",
            "criteria": {
                "true": "The answer states the same key facts as the reference "
                "(wording may differ)",
                "false": "The answer omits or contradicts a key fact in the reference",
            },
        }
    return qs


def build_state(row: dict) -> dict:
    evidence = "\n".join(c for c in row["contexts"] if c)[:MAX_EVIDENCE_CHARS]
    state = {"question": row["question"], "evidence": evidence, "answer": row["answer"]}
    if row.get("reference"):
        state["reference_answer"] = row["reference"]
    return state


def post_verdicts(state: dict, questions: dict, base_url: str, api_key: str) -> dict:
    """POST /v1/verdicts with backoff on 429/502/503/504 (fair-queue behaviour)."""
    import httpx

    delay = 5.0
    for attempt in range(6):
        resp = httpx.post(
            base_url.rstrip("/") + "/verdicts",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"state": state, "questions": questions, "on_no_answer": "low_confidence"},
            timeout=600,
        )
        if resp.status_code == 200:
            return resp.json()  # type: ignore[no-any-return]
        if resp.status_code in (429, 502, 503, 504) and attempt < 5:
            time.sleep(delay)
            delay *= 2
            continue
        raise RuntimeError(f"Verdicts HTTP {resp.status_code}: {resp.text[:200]}")
    raise RuntimeError("Verdicts: retries exhausted")


def p_true(verdict: dict) -> float | None:
    if verdict.get("answered") is False:
        return None
    return float(verdict["probabilities"]["true"])


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def is_abstained(answer: str) -> bool:
    return answer.strip().startswith(ABSTAIN_PREFIXES)


def classify(row: dict, p_grounded: float | None) -> str:
    if row.get("error") or not row["answer"].strip():
        return "error"
    if is_abstained(row["answer"]):
        return "abstained"
    if p_grounded is None:
        return "unsure"
    return "supported" if p_grounded >= SUPPORTED_P else "unsupported"


def score_row(row: dict, verdicts: dict | None) -> dict:
    p_g = p_a = p_m = None
    if verdicts:
        p_g = p_true(verdicts["grounded"])
        p_a = p_true(verdicts["answers_question"])
        if "matches_reference" in verdicts:
            p_m = p_true(verdicts["matches_reference"])
    outcome = classify(row, p_g)
    correct = (
        outcome == "supported"
        and p_a is not None
        and p_a >= MATCH_P
        and (not row.get("reference") or (p_m is not None and p_m >= MATCH_P))
    )
    return {
        "id": row["question_id"],
        "category": row["category"],
        "outcome": outcome,
        "correct": correct,
        "correct_high_confidence": correct and p_g is not None and p_g >= CONFIDENT_P,
        "p_grounded": p_g,
        "p_answers_question": p_a,
        "p_matches_reference": p_m,
        "pipeline_verified": row["is_verified"],
        "intent_correct": row["intent_correct"],
        "latency_ms": round(row["latency_ms"]),
        "has_reference": bool(row.get("reference")),
    }


def summarize(scored: list[dict]) -> dict:
    def block(items: list[dict]) -> dict:
        n = len(items)
        outcomes = Counter(s["outcome"] for s in items)
        refs = [s for s in items if s["has_reference"]]
        return {
            "n": n,
            "correct": sum(s["correct"] for s in items),
            "correct_high_confidence": sum(s["correct_high_confidence"] for s in items),
            "outcomes": {
                k: outcomes.get(k, 0)
                for k in ("supported", "unsure", "unsupported", "abstained", "error")
            },
            "pipeline_verified": sum(s["pipeline_verified"] for s in items),
            "intent_correct": sum(s["intent_correct"] for s in items),
            "reference_questions": len(refs),
            "reference_correct": sum(s["correct"] for s in refs),
        }

    by_cat = {
        c: block([s for s in scored if s["category"] == c])
        for c in sorted({s["category"] for s in scored})
    }
    return {"overall": block(scored), "by_category": by_cat}


# --------------------------------------------------------------------------
# Generation (checkpointed) + judging
# --------------------------------------------------------------------------


def generate_rows(questions: list[dict], args: argparse.Namespace, checkpoint: Path) -> list[dict]:
    from scripts.run_eval import collect_eval_rows

    done: dict[str, dict] = {}
    if checkpoint.exists() and not args.regenerate:
        done = {r["question_id"]: r for r in json.loads(checkpoint.read_text(encoding="utf-8"))}
        logger.info("Resuming: %d/%d questions already generated.", len(done), len(questions))

    for q in questions:
        qid = q["id"]
        if qid in done and not done[qid].get("error"):
            continue
        row = asdict(collect_eval_rows([q], args.index_dir, args.db_path, args.kg_path)[0])
        done[qid] = row
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps(list(done.values()), indent=2), encoding="utf-8")
        logger.info("Generated %s (%s) error=%s", qid, row["category"], row["error"])
    return [done[q["id"]] for q in questions if q["id"] in done]


def _format_sql_rows(columns: list[str], rows: list) -> list[str]:
    return [" | ".join(f"{c}: {v}" for c, v in zip(columns, r)) for r in rows]


def with_full_sql_evidence(row: dict, db_path: Path | None) -> dict:
    """Replace the pipeline's 10-row SQL evidence with the rows the narration saw."""
    sql = row.get("sql_query") or ""
    if not sql or db_path is None:
        return row
    from procurement_copilot.sql.safe_sql import execute_safe_sql

    result = execute_safe_sql(sql, db_path=db_path)
    if result.error:
        return row
    old = set(_format_sql_rows(result.columns, result.rows[:10]))
    kept = [c for c in row["contexts"] if c not in old]
    full = _format_sql_rows(result.columns, result.rows[:SQL_EVIDENCE_ROWS])
    return {**row, "contexts": kept + full}


def judge_rows(
    rows: list[dict], base_url: str, api_key: str, db_path: Path | None = None
) -> list[dict]:
    def one(row: dict) -> dict:
        if row.get("error") or not row["answer"].strip() or is_abstained(row["answer"]):
            return score_row(row, None)  # nothing to judge
        row = with_full_sql_evidence(row, db_path)
        verdicts = post_verdicts(
            build_state(row), build_questions(bool(row.get("reference"))), base_url, api_key
        )["verdicts"]
        return score_row(row, verdicts)

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        return list(pool.map(one, rows))


def print_summary(summary: dict) -> None:
    o = summary["overall"]
    print("\n" + "=" * 64)
    print("INDEPENDENT-JUDGE EVALUATION (ASU RC Verdicts API)")
    print("=" * 64)
    print(f"  Judged correct:      {o['correct']} / {o['n']}")
    print(f"    high-confidence:   {o['correct_high_confidence']} / {o['n']}  (p(grounded) >= 0.8)")
    print(f"  Outcomes:            {o['outcomes']}")
    print(f"  Pipeline self-check: {o['pipeline_verified']} / {o['n']} marked verified")
    print(f"  Intent routing:      {o['intent_correct']} / {o['n']}")
    print(
        f"  Reference subset:    {o['reference_correct']} / {o['reference_questions']} "
        "correct vs reference"
    )
    for cat, b in summary["by_category"].items():
        print(f"  - {cat:6} {b['correct']:2}/{b['n']:2} correct  {b['outcomes']}")


def configure_environment() -> None:
    """Mirror the production configuration before the package reads its settings
    (done in main(), not at import, so importing this module has no side effects)."""
    for key, value in {
        "LLM_PROVIDER": "groq",
        "LLM_FALLBACK_PROVIDER": "gemini",
        "EMBEDDING_PROVIDER": "fastembed",
        "RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "LANGFUSE_PUBLIC_KEY": "",
        "LANGFUSE_SECRET_KEY": "",
        # The Docker image copies vector_index_cloud to vector_index; pointing the setting at it
        # keeps retrieval (incl. FAR section pinning, which keys off this setting) identical.
        "VECTOR_INDEX_DIR": str(_DEFAULT_INDEX),
    }.items():
        os.environ.setdefault(key, value)
    if not hasattr(os, "sched_getaffinity"):  # Linux-only call used by the fastembed helpers
        os.sched_getaffinity = lambda pid: set(range(os.cpu_count() or 1))  # type: ignore[attr-defined]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def main() -> None:
    configure_environment()
    parser = argparse.ArgumentParser(description="Independent-judge evaluation")
    parser.add_argument(
        "--questions", type=Path, default=_PROJECT_ROOT / "eval" / "gold_questions.jsonl"
    )
    parser.add_argument("--results-dir", type=Path, default=_PROJECT_ROOT / "eval" / "results")
    parser.add_argument(
        "--index-dir", type=Path, default=None, help="default: settings.VECTOR_INDEX_DIR"
    )
    parser.add_argument(
        "--db-path", type=Path, default=_PROJECT_ROOT / "data" / "processed" / "procurement.duckdb"
    )
    parser.add_argument(
        "--kg-path", type=Path, default=_PROJECT_ROOT / "data" / "processed" / "kg.parquet"
    )
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions (0 = all)")
    parser.add_argument(
        "--regenerate", action="store_true", help="ignore the generation checkpoint"
    )
    args = parser.parse_args()

    from procurement_copilot.config import settings

    args.index_dir = args.index_dir or settings.VECTOR_INDEX_DIR
    if not (settings.RC_BASE_URL and settings.RC_API_KEY):
        sys.exit("RC_BASE_URL / RC_API_KEY are not set (put them in .env).")

    questions = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        questions = questions[: args.limit]
    logger.info("Evaluating %d questions.", len(questions))

    rows = generate_rows(questions, args, args.results_dir / "_judge_rows.json")
    scored = judge_rows(rows, settings.RC_BASE_URL, settings.RC_API_KEY, args.db_path)
    summary = summarize(scored)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.results_dir / f"judge_{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "judge": "ASU RC Verdicts API (jeffan-31b-1710)",
                "generator": (
                    f"{os.environ['LLM_PROVIDER']} + fallback "
                    f"{os.environ['LLM_FALLBACK_PROVIDER']}"
                ),
                "thresholds": {
                    "supported": SUPPORTED_P,
                    "confident": CONFIDENT_P,
                    "match": MATCH_P,
                },
                "summary": summary,
                "per_question": scored,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print_summary(summary)
    print(f"  Results: {out}")


if __name__ == "__main__":
    main()
