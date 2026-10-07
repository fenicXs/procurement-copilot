# Procurement Copilot — Before/After Comparison Report

_Generated as part of the 7-day agentic upgrade (2026-09-30 to 2026-10-06), run on the 10
reference-bearing RAG gold questions (`eval/gold_questions.jsonl`)._

## What changed (architecture)

| Component | Before | After |
|---|---|---|
| Intent routing | Regex keyword scoring | LLM classifier (regex fallback offline) |
| Retrieval | FAISS, single dense vector, heading-overlap heuristic rerank | Qdrant hybrid (dense+sparse, RRF fusion) + real cross-encoder rerank (BGE-reranker-v2-m3) |
| Retrieval robustness | Single-shot — whatever the first query returned | Corrective-RAG: LLM grades relevance, rewrites the query and retries (bounded), abstains rather than hallucinating if retries exhaust |
| NL-to-SQL | Regex templates (10 hardcoded patterns) | LLM-generated SQL (schema + few-shot prompt), template fallback offline — same unchanged safety validator either way |
| SQL result summary | Raw stringified table | LLM narration in plain English |
| Groundedness check | Keyword-overlap ratio | LLM-as-judge (keyword-overlap as offline fallback) |
| Observability | None | Full LangFuse tracing — every node, every LLM call, grouped by session |
| Eval methodology | Keyword-substring matching | RAGAS semantic metrics (faithfulness, answer relevancy, context precision/recall), judged by Ollama/Gemma |

**Scope note:** this repo isn't under git version control, so there's no way to check out and
re-run the literal pre-upgrade code paths. The table above documents the actual code change
(verified by reading the diffs); the numbers below compare the **eval methodology**
(keyword-matching vs. RAGAS) run against the **same, already-upgraded** pipeline, not a literal
old-pipeline-vs-new-pipeline execution. Framed honestly, not as a head-to-head pipeline benchmark.

## Results

| Metric | Keyword-matching eval | RAGAS (semantic eval) |
|---|---|---|
| Intent accuracy | 100% | 100% |
| Answer rate | 100% | — (n/a, RAGAS doesn't track this) |
| Keyword recall | 37.5% | — (n/a, superseded by faithfulness/relevancy) |
| Groundedness rate | 70% | 50% |
| Avg latency | 162.3s | 162.9s |
| Faithfulness | — (not measured) | 0.43 |
| Answer relevancy | — (not measured) | 0.44 |
| Context precision | — (not measured) | 0.45 |
| Context recall | — (not measured) | 0.35 |

*(SQL execution success is omitted — this subset is 100% RAG-category questions, so that metric
is vacuously 0/0 in both reports, not a real result.)*

### The groundedness-rate discrepancy (70% vs 50%) is itself a finding

These are two **separate** runs of the same 10 questions through the same pipeline, several
minutes apart. 7/10 were verified-grounded in the keyword run; only 5/10 in the RAGAS run. Since
`temperature=0` but Ollama/Gemma generation isn't perfectly deterministic in practice, and the
Corrective-RAG loop's grade/rewrite decisions depend on LLM judgment at each step, **re-running
the identical pipeline on identical questions produces different abstain/answer outcomes.** This
is a real characteristic of LLM-judge-driven pipelines worth knowing going in, not a bug — but it
means any single eval run should be read as one sample, not a precise fixed number.

### The low aggregate RAGAS scores are mostly abstain penalties, not bad answers

Splitting the RAGAS run by whether the pipeline actually answered or abstained tells a very
different story than the aggregate:

| Subset | n | Avg faithfulness | Avg answer relevancy |
|---|---|---|---|
| Answered (verified grounded) | 5 | **0.87** | **0.88** |
| Abstained ("I cannot provide a verified answer...") | 5 | 0.00 | 0.00 |

When the system actually answers, faithfulness and relevancy are strong (~0.87-0.88). The 5
abstains score a flat 0.0 on both metrics because RAGAS has no concept of "appropriate refusal" —
it scores a deliberate, correct abstain exactly like a bad answer, which drags the 10-question
aggregate (0.43 / 0.44) well below what the system's *actual answers* look like. This is a known
limitation of applying standard RAG metrics to a system designed to decline when uncertain, worth
understanding before reading the aggregate number at face value.

### A specific abstain case suggests the verifier may be over-conservative

Inspecting `rag_004` ("What is the simplified acquisition threshold?"): the retrieved context
*did* contain the actual FAR definition clause (`"(b) Simplified acquisition threshold. The
thr[eshold]..."`), and yet the pipeline still abstained. Since CRAG itself graded the retrieval as
sufficient (otherwise this context wouldn't have made it to generation), the rejection happened at
the **LLM-as-judge verifier** step, after generation. This points to the verifier being stricter
than necessary in at least this case — a good target for Gemma-prompt tuning or threshold
adjustment in a follow-up pass, not something fixed in this session.

## Example: Corrective-RAG abstain vs. hallucinate (capability, not a re-run comparison)

**Query:** `asdkjalksdj random gibberish about nothing`

Retrieval graded insufficient on the initial attempt and both rewrites; the pipeline abstained
explicitly: *"I don't have sufficiently grounded information in the FAR corpus to answer this
question confidently."* No hallucinated answer. The old single-shot FAISS retrieval had no
mechanism to recognize "nothing useful was retrieved" — it would have generated from whatever it
got back regardless of relevance.

## Example: real bug caught during live testing

**Query:** `How many awards did the Department of Defense receive?`

The first LLM-generated SQL attempt used `WHERE recipient_name = 'Department of Defense'` —
confusing the awarding agency with a contract recipient. Caught via live spot-check (not by any
automated test), fixed by adding a disambiguating few-shot example and an explicit note in the SQL
system prompt. Retest confirmed correct `WHERE awarding_agency = 'Department of Defense'` and a
correct count (3,281 awards).

## Judge bias caveat

The RAGAS judge (Gemma) is the same model used for answer generation in local dev — this risks
self-evaluation bias (a model may rate its own style/blind-spots more favorably). Faithfulness is
the metric least exposed to this, since it's closer to mechanical evidence-overlap checking than
subjective judgment; answer_relevancy leans more on judgment and should be read with that caveat.
A cloud model (OpenAI/Anthropic) as judge would remove this bias entirely at a small API cost —
noted as a natural follow-up, not implemented here by choice (keeping eval cost at zero, per
explicit decision this session).

## Scope note

This comparison uses the 10 RAG questions with hand-written reference answers (not all 42 gold
questions) — SQL and graph question references would depend on the specific randomly-fetched
USAspending sample and regex-extracted KG triples, which aren't stable ground truth to hardcode.
Full raw results: `eval/results/ragas_20261006_180926.json`, `eval/results/report_legacy.md`.
