# Failure analysis: 18 non-correct eval questions (run judge_20261107_172803)

Method: 6 independent sonnet auditors (2 each: retrieval, generation/verifier, gold-question) read per-question evidence bundles (`eval/results/failure_analysis/<id>.md`, built by `scripts/failure_bundle.py`). Labels below are the majority; votes are out of 6. Corpus facts were then spot-checked directly against `data/processed/far_chunks.jsonl` and `kg.parquet`.

Caveat: auditors are LLMs reading bundles; none re-ran the pipeline. Treat labels as strong hypotheses until a fix is verified.

| id | Majority label | Votes | Notes |
|---|---|---|---|
| rag_002 | generation (incomplete) | 6/6 | Answer repeats the "6.401(a)" pointer; the four conditions were in evidence [3] |
| rag_004 | gold_bad | 6/6 | Reference says $250,000; FAR 2.101 in corpus says $350,000. Judge p_grounded=0.25 on a correct answer |
| rag_006 | verifier_overstrict | 5/6 (1 ranking) | Abstain unjustified: 6.303/6.304 text was in evidence |
| rag_007 | generation | 6/6 | Invented "management capability, schedule, innovation" |
| rag_009 | retrieval_miss | 5/6 | Only 6.302-1 reached the generator; 6.302-2..-7 exist in corpus |
| rag_010 | generation | 4/6 (2 judge) | Wrong paragraph citations; over-generalised the 18-month rule |
| rag_012 | retrieval_miss | 6/6 | Abstain justified. Chunk `far_chunk_00179` has "Micro-purchase threshold means $15,000"; gold ($10,000) is stale |
| rag_013 | retrieval_miss | 6/6 | Abstain justified. Part 43 exists in corpus (e.g. `far_chunk_04591`) but was not retrieved |
| rag_014 | judge_overstrict | 6/6 | p_matches_ref 0.85, grounded 0.15 |
| rag_015 | generation | 5/6 (1 retrieval) | Presents the 52.249-3 dismantling clause as the general TFC procedure; Part 49 exists (`far_chunk_05006`) but absent from evidence |
| rag_020 | judge_overstrict | 6/6 | p_matches_ref 0.81 |
| sql_010 | generation | 6/6 | Narration invents "6 to 12 awards" for agencies not in the 10 rows shown |
| graph_003 | retrieval_miss | 5/6 (1 gold_bad) | Generic relation question, no anchor entity; graph has 82,827 `references` triples |
| graph_004 | routing | 6/6 | "show the triples" went to rag; answer then mislabelled 42.1503 factors |
| graph_008 | verifier_overstrict | 6/6 | 100 relevant triples retrieved, still abstained |
| graph_009 | retrieval_miss | 6/6 | Predicate "authorizes" has 3,208 triples; entity-only lookup returned none |
| mixed_001 | SQL generation | split label, same fix | `description ILIKE '%FAR%'` pollutes the filter; abstain justified |
| mixed_002 | judge_overstrict | 6/6 | Content supported; answer has duplicated/contradictory sections and `far_chunk_*` cites |

## Counts by primary cause
- Generation (5): rag_002, rag_007, rag_010, rag_015, sql_010
- Retrieval (5): rag_009, rag_012, rag_013, graph_003, graph_009
- Judge over-strict (3): rag_014, rag_020, mixed_002
- Verifier over-strict (2): rag_006, graph_008
- Routing (1): graph_004
- Gold bad (1): rag_004 (and rag_012's reference is also stale)
- SQL generation (1): mixed_001

## Candidate fix groups (not yet approved or implemented)
| # | Fix | Ids | Likely file(s) |
|---|---|---|---|
| F1 | Graph retriever: predicate-name lookup + fallback sampling when no entity | graph_003, 009 (008 ranking) | graph retriever module |
| F2 | Verifier: accept triple-listing answers; relax abstain gate when partial definitional evidence exists | graph_008, rag_006 | `orchestrator/verifier.py` |
| F3 | Router: "triples/graph" phrasing routes to graph | graph_004 | `prompts/intent.txt` / regex |
| F4 | RAG retrieval: definition boost ("X means $"), Part-number/topic expansion, multi-section expansion for 6.302 | rag_009, 012, 013, 015 | `rag/retriever.py` |
| F5 | Generation prompts: cite paragraph ids from chunk text, expand cross-references, no invented examples; SQL narration only on returned rows | rag_002, 007, 010, sql_010 | `prompts/rag_answer.txt`, `prompts/sql_summarize.txt` |
| F6 | Text-to-SQL: never turn a document name into a filter; use a structured sole-source field | mixed_001 | `prompts/sql_generate.txt` |
| F7 | Gold fixes: rag_004 -> $350,000, rag_012 -> $15,000, graph_003 anchored | rag_004, 012, 003 | `eval/gold_questions.jsonl` |

Judge-overstrict items (rag_014, rag_020, mixed_002) have no pipeline fix except cleaning mixed_002's duplicated output.

## Review and decisions (coordination)
- Antigravity declined (its lane is read-and-summarise only); Codex failed twice (model not supported on the ChatGPT account: gpt-5.5 and gpt-5.4). A fresh-context Claude reviewer (opus, read-only) gave the advisory opinion below; its code claims were checked by hand.
- **Shared blind spot (verified):** the judge gets evidence as one unnumbered string (`eval_judge.py:115`) while answers cite `[n]` / `far_chunk_*`, so citations cannot be checked. The 12k-char cut is NOT the cause for rag_020 / mixed_002 (their evidence is 4,999 / 8,545 chars). Graph triples are emitted twice each (`orchestrator/graph.py:176-177`). `verifier_node` auto-verifies SQL-only answers, so sql_010's invented narration is not caught by F5 alone.
- **Judge-overstrict:** option (b). First fix the harness (numbered evidence, de-duplicated triples), re-judge all 42 with frozen thresholds and report before/after. Then add a second judge model if available. Lowering the threshold after seeing results is rejected (p-hacking).
- **Noise baseline:** re-judge the unchanged checkpoint once to measure judge variance. At n=42 one question is 2.4 points.
- **Order:** harness fix -> F7+F3+F6 together -> F1 alone -> F2, F4, F5 each alone, each followed by a full 42-question re-run.
- **Graph:** fix the retriever (graph_009 is a real predicate-lookup gap); anchor graph_003 only as a logged gold edit.
- **Overfitting:** write the held-out set (10-15 questions) BEFORE any fix and score it separately. Audit all dollar/threshold values in the gold set, not only rag_004 / rag_012.

## Execution log
### Step 1: judge noise + evidence-format experiment (2026-11-07)
| Run (same 42 stored answers, same judge, same thresholds) | Correct | Notes |
|---|---|---|
| Original | 24/42 | baseline |
| Re-judged, same input (noise) | 24/42 | **0 flips**; hi-conf 14 -> 15. The judge is stable |
| Numbered evidence `[n]`, de-duplicated triples | 23/42 | +rag_007, -rag_016, -graph_006. Did NOT rescue rag_014/020/mixed_002 (p_grounded 0.14 / 0.12 / 0.35) |

Conclusion: the "judge cannot check citations" hypothesis is **not supported**; the three judge-overstrict cases stay as judge disagreement. Numbering is kept as opt-in (`--numbered-evidence`); the baseline format stays default so all before/after comparisons are like-for-like. Added `--judge-only` (re-judge the checkpoint, no Groq), `--gold` alias, `--tag`. Tests: 16 pass.

### Step 2: gold audit
- Only two dollar values existed in the gold set; both were stale vs the corpus and are fixed: rag_004 $250,000 -> $350,000; rag_012 $10,000 -> $15,000 (`far_chunk_00179`).
- **New finding:** the `awards` table has no competition / sole-source column (columns: award_id, recipient_name, dates, award_amount, outlays, description, covid/infra fields, awarding_agency, awarding_sub_agency, contract_award_type, ...). So mixed_001's "how much has been spent on sole-source" cannot be answered reliably from the DB; the auditors' suggested `extent_competed` filter does not exist. Correct system behaviour is to answer the FAR half and say the DB cannot identify sole-source awards. F6 is therefore "stop injecting `%FAR%`; state the limitation", not "use a competition field".
- graph_003 left unchanged until the graph retriever fix (F1) so that the edit is logged separately.

### Step 3: held-out set + baselines (2026-11-07, before any pipeline fix)
- `eval/gold_heldout.jsonl`: 15 questions (6 rag from Parts 8/9/12/15/28/42, 6 sql with filters/aggregates, 2 graph entity-anchored, 1 mixed); every reference sourced from corpus chunks or computed from the DB and spot-checked.
- **Baselines (all with the same judge, thresholds and baseline evidence format):**

| Set | Correct | Notes |
|---|---|---|
| Dev (42), corrected gold | **25/42** (16 hi-conf) | +1 vs 24: rag_004 now matches its corrected reference |
| Held-out (15) | **8/15** | rag 5/6, sql 3/6, graph 0/2, mixed 0/1. Same weak categories as dev |

- First held-out run had 3 pipeline errors (Groq daily-token 429; the Gemini fallback did not rescue them, likely Gemini per-minute limits in a burst, unconfirmed). Re-run succeeded.
- Held-out failures: heldout_sql_006 is a **real SQL bug** ("NASA" not mapped to "National Aeronautics and Space Administration", so the query returned Humana); heldout_rag_005 partial answer on 42.302 (retrieval); heldout_sql_003/005 have correct counts (2,261 / 614) but the judge says unsupported (single-number evidence, plus an invented gloss in sql_003); graph 0/2 and mixed 0/1 unsupported.
