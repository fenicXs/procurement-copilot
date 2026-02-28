"""Tests for the LangGraph orchestrator: intent routing, nodes, and verifier."""

from pathlib import Path

import duckdb
import pytest
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

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

# --- Fixtures ---


@pytest.fixture()
def tmp_faiss_index(tmp_path: Path) -> Path:
    """Build a small FAISS index for orchestrator tests."""
    docs = [
        Document(
            page_content=(
                "The contracting officer must obtain approval from the agency head "
                "for sole-source contracts exceeding the simplified acquisition threshold."
            ),
            metadata={
                "chunk_id": "far_chunk_00001",
                "page_start": 10,
                "page_end": 11,
                "section_heading": "6.302-1",
                "source_url": "https://www.acquisition.gov/far",
            },
        ),
        Document(
            page_content=(
                "Small business set-asides are required when the anticipated "
                "contract value exceeds $250,000."
            ),
            metadata={
                "chunk_id": "far_chunk_00002",
                "page_start": 42,
                "page_end": 43,
                "section_heading": "19.502-2",
                "source_url": "https://www.acquisition.gov/far",
            },
        ),
    ]
    embeddings = FakeEmbeddings(size=1536)
    vs = FAISS.from_documents(docs, embeddings)
    index_dir = tmp_path / "test_index"
    vs.save_local(str(index_dir))
    return index_dir


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
    def test_graph_compiles(self, tmp_faiss_index: Path, tmp_db: Path, tmp_kg: Path) -> None:
        app = build_graph(
            index_dir=tmp_faiss_index,
            db_path=tmp_db,
            kg_path=tmp_kg,
        )
        assert app is not None


class TestRunQuery:
    def test_rag_query(self, tmp_faiss_index: Path, tmp_db: Path, tmp_kg: Path) -> None:
        result = run_query(
            "What does FAR 6.302 say about sole-source?",
            index_dir=tmp_faiss_index,
            db_path=tmp_db,
            kg_path=tmp_kg,
        )
        assert isinstance(result, CopilotResponse)
        assert result.intent == INTENT_RAG
        assert len(result.answer) > 0

    def test_sql_query(self, tmp_faiss_index: Path, tmp_db: Path, tmp_kg: Path) -> None:
        result = run_query(
            "How much total spending by agency?",
            index_dir=tmp_faiss_index,
            db_path=tmp_db,
            kg_path=tmp_kg,
        )
        assert isinstance(result, CopilotResponse)
        assert result.intent == INTENT_SQL
        assert len(result.answer) > 0

    def test_graph_query(self, tmp_faiss_index: Path, tmp_db: Path, tmp_kg: Path) -> None:
        result = run_query(
            "Show the graph triples connected to this entity",
            index_dir=tmp_faiss_index,
            db_path=tmp_db,
            kg_path=tmp_kg,
        )
        assert isinstance(result, CopilotResponse)
        assert result.intent == INTENT_GRAPH
        assert len(result.answer) > 0

    def test_response_has_verification(
        self, tmp_faiss_index: Path, tmp_db: Path, tmp_kg: Path
    ) -> None:
        result = run_query(
            "How many awards are there?",
            index_dir=tmp_faiss_index,
            db_path=tmp_db,
            kg_path=tmp_kg,
        )
        assert isinstance(result.is_verified, bool)
