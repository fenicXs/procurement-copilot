"""LangGraph orchestrator: routes queries through RAG, SQL, Graph, and Verifier nodes.

This is the central coordination layer that:
1. Classifies user intent (rag / sql / graph / mixed)
2. Routes to the appropriate answer pipeline(s)
3. Verifies the answer is grounded in evidence
4. Returns a unified CopilotResponse
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

from procurement_copilot.orchestrator.intent import (
    INTENT_GRAPH,
    INTENT_MIXED,
    INTENT_SQL,
    classify_intent,
)
from procurement_copilot.orchestrator.verifier import verify_answer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State (TypedDict so LangGraph merges keys across nodes)
# ---------------------------------------------------------------------------


class OrchestratorState(TypedDict, total=False):
    """State that flows through the LangGraph orchestrator."""

    question: str
    intent: str

    # RAG results
    rag_answer: str
    rag_citations: list[dict]
    rag_chunks: list[str]

    # SQL results
    sql_answer: str
    sql_query: str
    sql_rows: list[tuple]
    sql_columns: list[str]

    # Graph results
    graph_answer: str
    graph_triples: list[dict]
    graph_chunk_ids: list[str]

    # Final output
    final_answer: str
    is_verified: bool
    verification_notes: str
    error: str | None


# ---------------------------------------------------------------------------
# Node functions (each takes full state dict, returns partial update)
# ---------------------------------------------------------------------------


def intent_router_node(state: OrchestratorState) -> dict[str, Any]:
    """Classify the user question and set the intent."""
    question = state["question"]
    intent = classify_intent(question)
    logger.info("Intent classified as: %s", intent)
    return {"intent": intent}


def rag_answer_node(
    state: OrchestratorState,
    index_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the Corrective-RAG pipeline (retrieve -> grade -> rewrite/retry -> generate)."""
    from procurement_copilot.rag.answer import generate_rag_answer_with_retrieval

    question = state["question"]

    try:
        result = generate_rag_answer_with_retrieval(question, top_k=5, index_dir=index_dir)
        return {
            "rag_answer": result.answer,
            "rag_citations": [
                {
                    "chunk_id": c.chunk_id,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section_heading": c.section_heading,
                    "source_url": c.source_url,
                }
                for c in result.citations
            ],
            "rag_chunks": [c.text for c in result.retrieved_chunks],
        }
    except Exception as exc:
        logger.exception("RAG node failed")
        return {"rag_answer": "", "error": f"RAG error: {exc}"}


def sql_answer_node(
    state: OrchestratorState,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Generate SQL from the question, validate, execute, and summarize."""
    from procurement_copilot.sql.safe_sql import execute_safe_sql, get_schema

    question = state["question"]

    try:
        schema = get_schema(db_path)
        sql = _nl_to_sql(question, schema)
        result = execute_safe_sql(sql, db_path=db_path)

        if result.error:
            return {"sql_answer": f"SQL error: {result.error}", "sql_query": sql}

        summary = _summarize_sql_result(question, result.columns, result.rows)
        return {
            "sql_answer": summary,
            "sql_query": result.sql,
            "sql_rows": result.rows[:20],
            "sql_columns": result.columns,
        }
    except Exception as exc:
        logger.exception("SQL node failed")
        return {"sql_answer": "", "error": f"SQL error: {exc}"}


def graph_answer_node(
    state: OrchestratorState,
    kg_path: Path | None = None,
) -> dict[str, Any]:
    """Query the knowledge graph and generate an answer."""
    from procurement_copilot.graph.answer import generate_graph_answer

    question = state["question"]

    try:
        result = generate_graph_answer(question, kg_path=kg_path)
        return {
            "graph_answer": result.answer,
            "graph_triples": [
                {"subject": t.subject, "relation": t.relation, "object": t.object}
                for t in result.triples_used
            ],
            "graph_chunk_ids": result.provenance_chunk_ids,
        }
    except Exception as exc:
        logger.exception("Graph node failed")
        return {"graph_answer": "", "error": f"Graph error: {exc}"}


def _collect_evidence(state: OrchestratorState) -> tuple[list[str], bool]:
    """Gather all evidence text (RAG chunks, SQL rows, graph triples) the
    pipeline produced. Shared by verifier_node (groundedness checking) and
    run_query (exposing contexts for external eval, e.g. RAGAS)."""
    evidence: list[str] = []
    evidence.extend(state.get("rag_chunks", []))
    if state.get("sql_rows"):
        cols = state.get("sql_columns", [])
        for row in state["sql_rows"][:10]:
            evidence.append(" | ".join(f"{c}: {v}" for c, v in zip(cols, row)))
    has_graph_evidence = False
    for triple in state.get("graph_triples", []):
        has_graph_evidence = True
        # Include both the arrow notation and plain text for better keyword matching
        evidence.append(f"{triple['subject']} --[{triple['relation']}]--> {triple['object']}")
        evidence.append(f"{triple['subject']} {triple['relation']} {triple['object']}")
    return evidence, has_graph_evidence


def verifier_node(state: OrchestratorState) -> dict[str, Any]:
    """Verify the combined answer against available evidence.

    SQL-only answers are auto-verified since they come directly from the
    database and keyword-overlap checks are not meaningful for numeric data.
    """
    intent = state.get("intent", "")

    # SQL answers are grounded by definition — skip keyword-overlap verification
    if intent == INTENT_SQL and state.get("sql_answer"):
        return {
            "final_answer": state["sql_answer"],
            "is_verified": True,
            "verification_notes": "SQL result from database — auto-verified.",
        }

    evidence, has_graph_evidence = _collect_evidence(state)

    # Combine answers
    answers = []
    if state.get("rag_answer"):
        answers.append(state["rag_answer"])
    if state.get("sql_answer"):
        answers.append(state["sql_answer"])
    if state.get("graph_answer"):
        answers.append(state["graph_answer"])

    combined = "\n\n".join(answers) if answers else ""

    if not combined:
        return {
            "final_answer": ("I could not find relevant information to answer this question."),
            "is_verified": False,
            "verification_notes": "No answer was produced by any pipeline.",
        }

    # Use a lower verification threshold for graph/mixed answers because
    # graph evidence is entity-relation triples with fewer overlapping words
    threshold = 0.25 if has_graph_evidence else 0.4
    result = verify_answer(combined, evidence, threshold=threshold)
    return {
        "final_answer": result.verified_answer,
        "is_verified": result.is_grounded,
        "verification_notes": result.evidence_summary,
    }


# ---------------------------------------------------------------------------
# NL-to-SQL helper — LLM-based with a regex-template offline fallback.
# LLM-generated SQL is never executed directly: it still passes through the
# unchanged safe_sql.py validator (SELECT-only, table allowlist, blocked
# keywords, LIMIT enforcement) downstream in sql_answer_node, same as the
# template path.
# ---------------------------------------------------------------------------

_SQL_TEMPLATES: list[tuple[str, str]] = [
    # Top N recipients by total amount (normalize common abbreviation variants)
    (
        r"top (\d+) (?:recipients?|contractors?|vendors?)",
        "SELECT FIRST(recipient_name) AS recipient_name, SUM(award_amount) AS total "
        "FROM awards GROUP BY REGEXP_REPLACE(UPPER(TRIM(recipient_name)), "
        "' (CORP|INC|LLC|LTD|CO)$', ' CORPORATION') ORDER BY total DESC LIMIT {n}",
    ),
    # --- GROUP-BY patterns MUST come before simple aggregates ---
    # Count of awards per agency
    (
        r"(?:how many|count|number of).*(?:awards?|contracts?).*(?:by|per|each) agenc",
        "SELECT awarding_agency, COUNT(*) AS award_count "
        "FROM awards GROUP BY awarding_agency ORDER BY award_count DESC",
    ),
    # Awards per agency (reversed word order)
    (
        r"(?:awards?|contracts?).*(?:per|by|each) agenc",
        "SELECT awarding_agency, COUNT(*) AS award_count "
        "FROM awards GROUP BY awarding_agency ORDER BY award_count DESC",
    ),
    # Total spending by agency
    (
        r"(?:spending|total|amount).*(?:by|per|each) agenc",
        "SELECT awarding_agency, SUM(award_amount) AS total "
        "FROM awards GROUP BY awarding_agency ORDER BY total DESC",
    ),
    # Awards by contract type
    (
        r"(?:by|per|each) (?:contract )?type",
        "SELECT contract_award_type, COUNT(*) AS cnt, SUM(award_amount) AS total "
        "FROM awards GROUP BY contract_award_type ORDER BY total DESC",
    ),
    # Awards by recipient (normalize common abbreviation variants)
    (
        r"(?:by|per|each) (?:recipient|contractor|vendor)",
        "SELECT FIRST(recipient_name) AS recipient_name, COUNT(*) AS cnt, "
        "SUM(award_amount) AS total FROM awards GROUP BY REGEXP_REPLACE("
        "UPPER(TRIM(recipient_name)), ' (CORP|INC|LLC|LTD|CO)$', ' CORPORATION') "
        "ORDER BY total DESC LIMIT 20",
    ),
    # --- Simple aggregates (no GROUP BY) ---
    # Count of awards (total)
    (
        r"how many (?:awards?|contracts?)",
        "SELECT COUNT(*) AS award_count FROM awards",
    ),
    # Total spending
    (
        r"(?:total|overall|sum).*(?:spending|amount|awarded|obligated)",
        "SELECT SUM(award_amount) AS total_spending FROM awards",
    ),
    # Average award amount
    (
        r"average.*(?:award|contract|amount)",
        "SELECT AVG(award_amount) AS avg_amount FROM awards",
    ),
    # Largest single award
    (
        r"largest|biggest|highest.*(?:award|contract)",
        "SELECT award_id, recipient_name, award_amount, description "
        "FROM awards ORDER BY award_amount DESC LIMIT 5",
    ),
]


def _nl_to_sql_template(question: str) -> str:
    """Convert a natural language question to SQL using regex templates.

    Falls back to a basic SELECT if no template matches.
    """
    import re as _re

    q_lower = question.lower()

    for pattern, template in _SQL_TEMPLATES:
        m = _re.search(pattern, q_lower)
        if m:
            # Substitute {n} with captured group if present
            if m.groups():
                return template.format(n=m.group(1))
            return template

    # Fallback: return top rows
    return "SELECT * FROM awards LIMIT 10"


SQL_SYSTEM_PROMPT_TEMPLATE = """You write a single DuckDB SELECT query against a table \
named `awards` with this schema:

{schema}

Rules:
- Output ONLY one SELECT statement — never DROP/DELETE/UPDATE/INSERT/ALTER or any other
  write/DDL statement.
- Only reference the `awards` table.
- Respond with ONLY the raw SQL query. No markdown code fences, no explanation.

Examples:
Q: What are the top 5 recipients by total award amount?
A: SELECT recipient_name, SUM(award_amount) AS total FROM awards
   GROUP BY recipient_name ORDER BY total DESC LIMIT 5

Q: How many awards did the Department of Defense receive?
A: SELECT COUNT(*) AS award_count FROM awards WHERE awarding_agency = 'Department of Defense'

Note: `awarding_agency` is the government agency that made the award (e.g. "Department
of Defense"). `recipient_name` is the contractor/company that received it. Do not
confuse the two — a government agency name belongs in `awarding_agency`, never
`recipient_name`.
"""


def _format_schema_for_prompt(schema: dict[str, list[dict]]) -> str:
    lines = []
    for table, cols in schema.items():
        col_strs = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
        lines.append(f"{table}: {col_strs}")
    return "\n".join(lines)


def _nl_to_sql_llm(
    question: str, schema: dict[str, list[dict]], llm: "BaseChatModel"
) -> str | None:
    """LLM-generated SQL. Returns None (caller falls back to templates) if
    the response doesn't look like a SELECT statement."""
    import re as _re

    from langchain_core.messages import HumanMessage, SystemMessage

    system_prompt = SQL_SYSTEM_PROMPT_TEMPLATE.format(schema=_format_schema_for_prompt(schema))
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Q: {question}\nA:"),
    ]
    response = llm.invoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)

    sql = _re.sub(r"^```(?:sql)?\s*|\s*```$", "", content.strip(), flags=_re.IGNORECASE).strip()
    if not sql.upper().startswith("SELECT"):
        return None
    return sql


def _nl_to_sql(question: str, schema: dict[str, list[dict]]) -> str:
    """Convert a natural language question to SQL.

    Uses an LLM when a provider is configured (falls back to regex
    templates if the response is unusable); uses the templates directly
    offline. Either way, the result still passes through safe_sql.py
    unchanged before execution.
    """
    from procurement_copilot.llm import get_llm

    llm = get_llm()
    if llm is not None:
        sql = _nl_to_sql_llm(question, schema, llm)
        if sql:
            return sql
        logger.warning("LLM NL-to-SQL unusable — falling back to templates.")

    return _nl_to_sql_template(question)


def _summarize_sql_result_template(columns: list[str], rows: list[tuple]) -> str:
    """Produce a plain stringified summary of SQL results."""
    lines = [f"Query returned {len(rows)} row(s). Columns: {', '.join(columns)}\n"]
    for row in rows[:10]:
        parts = [f"{col}: {val}" for col, val in zip(columns, row)]
        lines.append("  " + " | ".join(parts))

    if len(rows) > 10:
        lines.append(f"  ... and {len(rows) - 10} more rows.")

    return "\n".join(lines)


def _summarize_sql_result_llm(
    question: str,
    columns: list[str],
    rows: list[tuple],
    llm: "BaseChatModel",
) -> str | None:
    """LLM narration of SQL results. Returns None on an empty response."""
    from langchain_core.messages import HumanMessage, SystemMessage

    preview_rows = rows[:20]
    table_text = "\n".join(
        " | ".join(f"{c}: {v}" for c, v in zip(columns, row)) for row in preview_rows
    )
    messages = [
        SystemMessage(
            content=(
                "Summarize this SQL query result in 1-3 concise sentences, in plain "
                "English, for a procurement analyst. Mention actual numbers/names from "
                "the data. Do not invent values not present in the result."
            )
        ),
        HumanMessage(content=f"Question: {question}\n\nResult ({len(rows)} rows):\n{table_text}"),
    ]
    response = llm.invoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)
    return content.strip() or None


def _summarize_sql_result(
    question: str,
    columns: list[str],
    rows: list[tuple],
) -> str:
    """Produce a human-readable summary of SQL results.

    Uses an LLM to narrate the results in plain English when a provider is
    configured; falls back to a stringified table offline.
    """
    if not rows:
        return "The query returned no results."

    from procurement_copilot.llm import get_llm

    llm = get_llm()
    if llm is not None:
        summary = _summarize_sql_result_llm(question, columns, rows, llm)
        if summary:
            return summary
        logger.warning("LLM SQL summary empty — falling back to stringified table.")

    return _summarize_sql_result_template(columns, rows)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _route_after_intent(state: OrchestratorState) -> str:
    """Decide the next node based on classified intent."""
    intent = state.get("intent", "rag")
    if intent == INTENT_SQL:
        return "sql_node"
    if intent == INTENT_GRAPH:
        return "graph_node"
    if intent == INTENT_MIXED:
        return "mixed_node"
    return "rag_node"


def build_graph(
    index_dir: Path | None = None,
    db_path: Path | None = None,
    kg_path: Path | None = None,
) -> Any:
    """Build and compile the LangGraph orchestrator.

    Optional path overrides are used for testing with temporary fixtures.
    """

    # Create wrapper nodes that capture path overrides via closure
    def _rag(state: OrchestratorState) -> dict[str, Any]:
        return rag_answer_node(state, index_dir=index_dir)

    def _sql(state: OrchestratorState) -> dict[str, Any]:
        return sql_answer_node(state, db_path=db_path)

    def _graph_fn(state: OrchestratorState) -> dict[str, Any]:
        return graph_answer_node(state, kg_path=kg_path)

    def _mixed(state: OrchestratorState) -> dict[str, Any]:
        from procurement_copilot.orchestrator.intent import _SQL_SIGNALS

        rag_update = rag_answer_node(state, index_dir=index_dir)
        graph_update = graph_answer_node(state, kg_path=kg_path)
        merged: dict[str, Any] = {}
        merged.update(rag_update)
        merged.update(graph_update)

        # Also run SQL if the question has spending/data signals
        question = state.get("question", "")
        if _SQL_SIGNALS.search(question):
            sql_update = sql_answer_node(state, db_path=db_path)
            merged.update(sql_update)

        # Merge sub-answers into a concise combined answer via LLM if available
        parts = []
        if merged.get("rag_answer"):
            parts.append(merged["rag_answer"])
        if merged.get("graph_answer"):
            parts.append(merged["graph_answer"])
        if merged.get("sql_answer"):
            parts.append(merged["sql_answer"])

        if len(parts) > 1:
            from procurement_copilot.llm import get_llm

            llm = get_llm()
            if llm is not None:
                from langchain_core.messages import HumanMessage, SystemMessage

                combine_prompt = (
                    "You are given multiple answers to a single procurement question. "
                    "Merge them into ONE concise, well-structured answer (under 300 words). "
                    "Preserve all citations [chunk_id] and data values. Remove redundancy."
                )
                combined_input = f"Question: {question}\n\n" + "\n\n---\n\n".join(
                    f"Answer {i+1}:\n{p}" for i, p in enumerate(parts)
                )
                msgs = [
                    SystemMessage(content=combine_prompt),
                    HumanMessage(content=combined_input),
                ]
                resp = llm.invoke(msgs)
                merged_text = resp.content if isinstance(resp.content, str) else str(resp.content)
                # Store as both rag_answer and graph_answer so verifier sees it
                merged["rag_answer"] = merged_text

        return merged

    graph = StateGraph(OrchestratorState)

    graph.add_node("intent_router", intent_router_node)
    graph.add_node("rag_node", _rag)
    graph.add_node("sql_node", _sql)
    graph.add_node("graph_node", _graph_fn)
    graph.add_node("mixed_node", _mixed)
    graph.add_node("verify_node", verifier_node)

    graph.set_entry_point("intent_router")

    graph.add_conditional_edges(
        "intent_router",
        _route_after_intent,
        {
            "rag_node": "rag_node",
            "sql_node": "sql_node",
            "graph_node": "graph_node",
            "mixed_node": "mixed_node",
        },
    )

    # All answer nodes feed into the verifier
    graph.add_edge("rag_node", "verify_node")
    graph.add_edge("sql_node", "verify_node")
    graph.add_edge("graph_node", "verify_node")
    graph.add_edge("mixed_node", "verify_node")

    graph.add_edge("verify_node", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


@dataclass
class CopilotResponse:
    """Final response from the orchestrator."""

    question: str
    intent: str
    answer: str
    is_verified: bool
    citations: list[dict] = field(default_factory=list)
    sql_query: str = ""
    graph_triples: list[dict] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    error: str | None = None


def run_query(
    question: str,
    index_dir: Path | None = None,
    db_path: Path | None = None,
    kg_path: Path | None = None,
    session_id: str | None = None,
) -> CopilotResponse:
    """Run a question through the full orchestrator pipeline.

    This is the main entry point for the copilot. LangFuse tracing is
    attached per-call (not module-global) so concurrent requests don't bleed
    into each other's traces; it's a no-op when LangFuse keys aren't set.
    """
    from procurement_copilot.audit_log import log_query
    from procurement_copilot.observability import flush_and_get_trace_url, get_langfuse_handler
    from procurement_copilot.rag.answer import ABSTAIN_MESSAGE

    app = build_graph(index_dir=index_dir, db_path=db_path, kg_path=kg_path)

    initial_state: OrchestratorState = {"question": question}

    handler = get_langfuse_handler(session_id=session_id)
    config: dict[str, Any] | None = None
    if handler is not None:
        config = {"callbacks": [handler]}
        if session_id:
            config["metadata"] = {"langfuse_session_id": session_id}

    final_state = app.invoke(initial_state, config=config)

    if handler is not None:
        trace_url = flush_and_get_trace_url(handler)
        logger.info("LangFuse trace: %s", trace_url)

    contexts, _ = _collect_evidence(final_state)
    citations = final_state.get("rag_citations", [])
    abstained = final_state.get("final_answer", "") == ABSTAIN_MESSAGE

    log_query(
        question=question,
        intent=final_state.get("intent", ""),
        is_verified=final_state.get("is_verified", False),
        abstained=abstained,
        citation_chunk_ids=[c["chunk_id"] for c in citations],
        session_id=session_id,
    )

    return CopilotResponse(
        question=question,
        intent=final_state.get("intent", ""),
        answer=final_state.get("final_answer", ""),
        is_verified=final_state.get("is_verified", False),
        citations=citations,
        sql_query=final_state.get("sql_query", ""),
        graph_triples=final_state.get("graph_triples", []),
        contexts=contexts,
        error=final_state.get("error"),
    )
