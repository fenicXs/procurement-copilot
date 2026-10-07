"""Tests for the LangGraph orchestrator: intent routing, nodes, and verifier."""

from pathlib import Path

import duckdb
import pytest

from procurement_copilot.orchestrator.graph import (
    CopilotResponse,
    build_graph,
    intent_router_node,
    run_query,
    verifier_node,
)
from procurement_copilot.orchestrator.intent import (
    INTENT_GRAPH,
    INTENT_RAG,
    INTENT_SQL,
    classify_intent,
)
from procurement_copilot.orchestrator.verifier import (
    VerificationResult,
    verify_answer,
)

from ._qdrant_fixtures import build_tmp_qdrant_index

# --- Fixtures ---


@pytest.fixture()
def tmp_qdrant_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a small local Qdrant index for orchestrator tests."""

    class _LengthReranker:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [float(len(text)) for _, text in pairs]

    monkeypatch.setattr(
        "procurement_copilot.rag.retriever._get_reranker", lambda: _LengthReranker()
    )

    chunks = [
        {
            "chunk_id": "far_chunk_00001",
            "text": (
                "The contracting officer must obtain approval from the agency head "
                "for sole-source contracts exceeding the simplified acquisition threshold."
            ),
            "page_start": 10,
            "page_end": 11,
            "section_heading": "6.302-1",
            "source_url": "https://www.acquisition.gov/far",
        },
        {
            "chunk_id": "far_chunk_00002",
            "text": (
                "Small business set-asides are required when the anticipated "
                "contract value exceeds $250,000."
            ),
            "page_start": 42,
            "page_end": 43,
            "section_heading": "19.502-2",
            "source_url": "https://www.acquisition.gov/far",
        },
    ]
    return build_tmp_qdrant_index(tmp_path, chunks)


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Create a temporary DuckDB with sample award data."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE awards (
            internal_id BIGINT,
            award_id VARCHAR,
            recipient_name VARCHAR,
            start_date DATE,
            end_date DATE,
            award_amount DOUBLE,
            total_outlays DOUBLE,
            description VARCHAR,
            covid_obligations DOUBLE,
            covid_outlays DOUBLE,
            infrastructure_obligations DOUBLE,
            infrastructure_outlays DOUBLE,
            awarding_agency VARCHAR,
            awarding_sub_agency VARCHAR,
            contract_award_type VARCHAR,
            recipient_id VARCHAR,
            awarding_agency_id INTEGER,
            agency_slug VARCHAR,
            generated_internal_id VARCHAR
        );
    """
    )
    con.execute(
        """
        INSERT INTO awards VALUES
        (1, 'AWD-001', 'ACME CORP', '2023-01-15', '2024-01-15',
         1000000.0, 500000.0, 'Test contract', 0, 0, 0, 0,
         'Department of Defense', 'Army', 'DEFINITIVE CONTRACT',
         'r1', 100, 'dod', 'gen-1'),
        (2, 'AWD-002', 'GLOBEX INC', '2023-06-01', '2024-06-01',
         2500000.0, 1200000.0, 'IT services', 0, 0, 0, 0,
         'Department of Health and Human Services', 'CDC', 'BPA CALL',
         'r2', 200, 'hhs', 'gen-2');
    """
    )
    con.close()
    return db_path


@pytest.fixture()
def tmp_kg(tmp_path: Path) -> Path:
    """Create a temporary KG parquet with sample triples."""
    kg_path = tmp_path / "test_kg.parquet"
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE triples (subject VARCHAR, relation VARCHAR, "
        "object VARCHAR, chunk_id VARCHAR)"
    )
    con.executemany(
        "INSERT INTO triples VALUES (?, ?, ?, ?)",
        [
            ("6.302 1", "authorizes", "sole source", "far_chunk_00100"),
            ("6.302 1", "requires", "contracting officer", "far_chunk_00100"),
            ("19.502 2", "requires", "small business", "far_chunk_00200"),
        ],
    )
    con.execute(f"COPY triples TO '{kg_path}' (FORMAT PARQUET)")
    con.close()
    return kg_path


# --- Intent classification tests ---


class TestClassifyIntent:
    def test_sql_intent(self) -> None:
        intent = classify_intent("How much total spending by agency?")
        assert intent == INTENT_SQL

    def test_rag_intent(self) -> None:
        intent = classify_intent("What does FAR 6.302 say about sole-source contracting?")
        assert intent == INTENT_RAG

    def test_graph_intent(self) -> None:
        intent = classify_intent("Show the graph triples connected to this entity")
        assert intent == INTENT_GRAPH

    def test_unknown_defaults_to_rag(self) -> None:
        intent = classify_intent("Hello world")
        assert intent == INTENT_RAG


class _StubLLM:
    """Returns one canned response per .invoke() call, in order."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages: list):
        from dataclasses import dataclass

        @dataclass
        class _Resp:
            content: str

        self.calls.append(messages)
        return _Resp(content=self._responses.pop(0))


class TestClassifyIntentLLM:
    """classify_intent() dispatches to the LLM when a provider is configured."""

    def test_uses_llm_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = _StubLLM(["sql"])
        monkeypatch.setattr("procurement_copilot.llm.get_llm", lambda: stub)

        intent = classify_intent("This wouldn't match any regex pattern at all")

        assert intent == INTENT_SQL
        assert len(stub.calls) == 1

    def test_falls_back_to_regex_on_unparseable_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = _StubLLM(["I'm not sure, maybe something else entirely"])
        monkeypatch.setattr("procurement_copilot.llm.get_llm", lambda: stub)

        intent = classify_intent("How much total spending by agency?")

        assert intent == INTENT_SQL  # regex fallback still gets this right


# --- Verifier tests ---


class TestVerifier:
    def test_grounded_answer(self) -> None:
        answer = "The contracting officer must obtain approval for sole-source contracts."
        evidence = ["The contracting officer must obtain approval for sole-source contracts."]
        result = verify_answer(answer, evidence)
        assert isinstance(result, VerificationResult)
        assert result.is_grounded

    def test_ungrounded_answer_abstains(self) -> None:
        answer = "Quantum computing is used for protein folding simulations."
        evidence = ["The contracting officer must obtain approval."]
        result = verify_answer(answer, evidence)
        assert not result.is_grounded
        assert len(result.unsupported_claims) > 0

    def test_empty_answer(self) -> None:
        result = verify_answer("", ["some evidence"])
        assert result.is_grounded

    def test_no_evidence(self) -> None:
        result = verify_answer("Some answer.", [])
        assert not result.is_grounded


# --- Intent router node test ---


class TestIntentRouterNode:
    def test_returns_intent(self) -> None:
        state = {"question": "How much was spent by the Department of Defense?"}
        result = intent_router_node(state)
        assert "intent" in result
        assert result["intent"] == INTENT_SQL


# --- Verifier node test ---


class TestVerifierNode:
    def test_verifier_with_rag_evidence(self) -> None:
        state = {
            "rag_answer": "The contracting officer must obtain approval.",
            "rag_chunks": ["The contracting officer must obtain approval for sole-source."],
            "sql_answer": "",
            "sql_rows": [],
            "sql_columns": [],
            "graph_answer": "",
            "graph_triples": [],
        }
        result = verifier_node(state)
        assert result["final_answer"]
        assert "is_verified" in result

    def test_verifier_with_no_answers(self) -> None:
        state = {
            "rag_answer": "",
            "rag_chunks": [],
            "sql_answer": "",
            "sql_rows": [],
            "sql_columns": [],
            "graph_answer": "",
            "graph_triples": [],
        }
        result = verifier_node(state)
        assert "could not find" in result["final_answer"].lower()


# --- Integration tests: build_graph + run_query ---


class TestBuildGraph:
    def test_graph_compiles(self, tmp_qdrant_index: Path, tmp_db: Path, tmp_kg: Path) -> None:
        app = build_graph(
            index_dir=tmp_qdrant_index,
            db_path=tmp_db,
            kg_path=tmp_kg,
        )
        assert app is not None


class TestRunQuery:
    def test_rag_query(self, tmp_qdrant_index: Path, tmp_db: Path, tmp_kg: Path) -> None:
        result = run_query(
            "What does FAR 6.302 say about sole-source?",
            index_dir=tmp_qdrant_index,
            db_path=tmp_db,
            kg_path=tmp_kg,
        )
        assert isinstance(result, CopilotResponse)
        assert result.intent == INTENT_RAG
        assert len(result.answer) > 0

    def test_sql_query(self, tmp_qdrant_index: Path, tmp_db: Path, tmp_kg: Path) -> None:
        result = run_query(
            "How much total spending by agency?",
            index_dir=tmp_qdrant_index,
            db_path=tmp_db,
            kg_path=tmp_kg,
        )
        assert isinstance(result, CopilotResponse)
        assert result.intent == INTENT_SQL
        assert len(result.answer) > 0

    def test_graph_query(self, tmp_qdrant_index: Path, tmp_db: Path, tmp_kg: Path) -> None:
        result = run_query(
            "Show the graph triples connected to this entity",
            index_dir=tmp_qdrant_index,
            db_path=tmp_db,
            kg_path=tmp_kg,
        )
        assert isinstance(result, CopilotResponse)
        assert result.intent == INTENT_GRAPH
        assert len(result.answer) > 0

    def test_response_has_verification(
        self, tmp_qdrant_index: Path, tmp_db: Path, tmp_kg: Path
    ) -> None:
        result = run_query(
            "How many awards are there?",
            index_dir=tmp_qdrant_index,
            db_path=tmp_db,
            kg_path=tmp_kg,
        )
        assert isinstance(result.is_verified, bool)


# --- NL-to-SQL and SQL summary: LLM path + template fallback ---


class TestNLToSQLLLM:
    def test_uses_llm_sql_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from procurement_copilot.orchestrator.graph import _nl_to_sql

        stub = _StubLLM(["SELECT recipient_name FROM awards LIMIT 5"])
        monkeypatch.setattr("procurement_copilot.llm.get_llm", lambda: stub)

        sql = _nl_to_sql("Who are the top recipients?", {"awards": []})

        assert sql == "SELECT recipient_name FROM awards LIMIT 5"
        assert len(stub.calls) == 1

    def test_strips_markdown_fences(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from procurement_copilot.orchestrator.graph import _nl_to_sql

        stub = _StubLLM(["```sql\nSELECT COUNT(*) FROM awards\n```"])
        monkeypatch.setattr("procurement_copilot.llm.get_llm", lambda: stub)

        sql = _nl_to_sql("How many awards?", {"awards": []})

        assert sql == "SELECT COUNT(*) FROM awards"

    def test_falls_back_to_template_on_non_select_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from procurement_copilot.orchestrator.graph import _nl_to_sql

        stub = _StubLLM(["I don't know how to write that query."])
        monkeypatch.setattr("procurement_copilot.llm.get_llm", lambda: stub)

        sql = _nl_to_sql("How many awards?", {"awards": []})

        assert sql.upper().startswith("SELECT")  # template fallback

    def test_offline_uses_template_directly(self) -> None:
        from procurement_copilot.orchestrator.graph import _nl_to_sql

        sql = _nl_to_sql("How many awards?", {"awards": []})

        assert sql == "SELECT COUNT(*) AS award_count FROM awards"


class TestSummarizeSQLResultLLM:
    def test_uses_llm_narration_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from procurement_copilot.orchestrator.graph import _summarize_sql_result

        stub = _StubLLM(["ACME Corp received the largest award, at $1,000,000."])
        monkeypatch.setattr("procurement_copilot.llm.get_llm", lambda: stub)

        summary = _summarize_sql_result(
            "Who got the largest award?",
            ["recipient_name", "award_amount"],
            [("ACME Corp", 1_000_000)],
        )

        assert summary == "ACME Corp received the largest award, at $1,000,000."

    def test_offline_falls_back_to_stringified_table(self) -> None:
        from procurement_copilot.orchestrator.graph import _summarize_sql_result

        summary = _summarize_sql_result(
            "Who got the largest award?",
            ["recipient_name", "award_amount"],
            [("ACME Corp", 1_000_000)],
        )

        assert "ACME Corp" in summary
        assert "Query returned 1 row(s)" in summary

    def test_no_rows_short_circuits_without_llm_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from procurement_copilot.orchestrator.graph import _summarize_sql_result

        stub = _StubLLM([])
        monkeypatch.setattr("procurement_copilot.llm.get_llm", lambda: stub)

        summary = _summarize_sql_result("Any awards?", ["recipient_name"], [])

        assert summary == "The query returned no results."
        assert stub.calls == []
