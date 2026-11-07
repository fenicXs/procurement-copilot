"""Build one evidence file per non-correct eval question for root-cause analysis.

Stdlib only; reads the generation checkpoint (what the pipeline actually saw and said),
the gold set and a judge results file. No LLM calls, no network.

Usage (from procurement-copilot/):
    python scripts/failure_bundle.py [--judge eval/results/judge_<stamp>.json]
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_CHUNK_CHARS = 1500


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def keyword_hits(keywords: list[str], contexts: list[str], answer: str) -> dict:
    blob = "\n".join(contexts).lower()
    ans = answer.lower()
    return {k: {"in_evidence": k.lower() in blob, "in_answer": k.lower() in ans} for k in keywords}


def section_mentions(question: str, corpus: list[dict]) -> list[str]:
    """Corpus chunk ids that start with a FAR section number named in the question."""
    out: list[str] = []
    for num in re.findall(r"\b\d{1,2}\.\d{3}(?:-\d+)?\b", question):
        for c in corpus:
            if re.search(rf"(?m)^\s*{re.escape(num)}\b", c.get("text", "")):
                out.append(f"{num}: {c.get('chunk_id') or c.get('id')}")
    return out[:6]


def render(row: dict, gold: dict, judge: dict, corpus: list[dict]) -> str:
    ctx = row["contexts"]
    lines = [
        f"# {row['question_id']} ({row['category']})",
        f"**Question:** {row['question']}",
        f"**Expected intent:** {gold.get('expected_intent')} | **actual:** {row['actual_intent']}",
        f"**Pipeline verified flag:** {row['is_verified']} | **error:** {row['error']}",
        "**Judge:** outcome={outcome} p_grounded={g} p_answers={a} p_matches_ref={m}".format(
            outcome=judge["outcome"],
            g=_fmt(judge["p_grounded"]),
            a=_fmt(judge["p_answers_question"]),
            m=_fmt(judge["p_matches_reference"]),
        ),
        "",
        "## Reference answer",
        row.get("reference") or "(none)",
        "",
        "## Pipeline answer",
        row["answer"],
        "",
        f"## Expected keywords: {gold.get('expected_keywords')}",
        json.dumps(keyword_hits(gold.get("expected_keywords") or [], ctx, row["answer"])),
    ]
    if row.get("sql_query"):
        lines += ["", "## SQL", row["sql_query"]]
    hits = section_mentions(row["question"], corpus) if corpus else []
    if hits:
        lines += ["", "## Corpus chunks that start with a section named in the question", *hits]
    lines += ["", f"## Evidence the pipeline used ({len(ctx)} items)"]
    for i, c in enumerate(ctx, 1):
        lines.append(f"### [{i}]\n{c[:MAX_CHUNK_CHARS]}")
    return "\n".join(lines) + "\n"


def _fmt(p: float | None) -> str:
    return "n/a" if p is None else f"{p:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", type=Path, default=ROOT / "eval/results/judge_20261107_172803.json")
    ap.add_argument("--rows", type=Path, default=ROOT / "eval/results/_judge_rows.json")
    ap.add_argument("--gold", type=Path, default=ROOT / "eval/gold_questions.jsonl")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data/processed/far_chunks.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "eval/results/failure_analysis")
    args = ap.parse_args()

    judged = {
        j["id"]: j for j in json.loads(args.judge.read_text(encoding="utf-8"))["per_question"]
    }
    rows = {r["question_id"]: r for r in json.loads(args.rows.read_text(encoding="utf-8"))}
    gold = {g["id"]: g for g in load_jsonl(args.gold)}
    corpus = load_jsonl(args.corpus) if args.corpus.exists() else []

    args.out.mkdir(parents=True, exist_ok=True)
    failed = [qid for qid, j in judged.items() if not j["correct"]]
    for qid in failed:
        (args.out / f"{qid}.md").write_text(
            render(rows[qid], gold[qid], judged[qid], corpus), encoding="utf-8"
        )
    (args.out / "_index.json").write_text(
        json.dumps({q: judged[q]["outcome"] for q in failed}, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(failed)} bundles to {args.out}")


if __name__ == "__main__":
    main()
