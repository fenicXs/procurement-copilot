"""Tests for the knowledge graph extraction, query, and answer modules."""

from pathlib import Path

import duckdb
import pytest

from procurement_copilot.graph.answer import (
    GraphAnswer,
    _extract_query_entities,
    generate_graph_answer,
)
from procurement_copilot.graph.query import (
    Triple,
    find_by_relation,
    find_related,
    get_entity_neighbors,
    get_provenance_chunk_ids,
    multi_hop,
)
from scripts.build_kg import extract_triples

# --- extract_triples unit tests ---


class TestExtractTriples:
    def test_extracts_section_concept_triples(self) -> None:
        text = (
            "FAR 6.302-1 allows sole source contracting when the contracting officer "
            "determines that only one responsible source exists."
        )
        triples = extract_triples("chunk_001", text)
        assert len(triples) > 0
        subjects = {t["subject"] for t in triples}
        objects = {t["object"] for t in triples}
        # Should find the section and the concept
        assert any("6.302" in s for s in subjects)
        assert any("sole source" in o or "sole-source" in o for o in objects)

    def test_extracts_role_triples(self) -> None:
        text = "The contracting officer shall ensure competition unless an exception applies."
        triples = extract_triples("chunk_002", text)
        all_entities = {t["subject"] for t in triples} | {t["object"] for t in triples}
        assert any("contracting officer" in e for e in all_entities)

    def test_cross_references(self) -> None:
        text = "See FAR 15.304 and FAR 15.305 for evaluation factors and evaluation process."
        triples = extract_triples("chunk_003", text)
        ref_triples = [t for t in triples if t["relation"] == "references"]
        assert len(ref_triples) > 0

    def test_empty_text_returns_empty(self) -> None:
        triples = extract_triples("chunk_empty", "No procurement terms here at all.")
        assert triples == []

    def test_provenance_preserved(self) -> None:
        text = "FAR 19.502-2 requires small business set-asides."
        triples = extract_triples("my_chunk", text)
        assert all(t["chunk_id"] == "my_chunk" for t in triples)


# --- Graph query tests with temp KG ---


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
            ("19.502 2", "references", "6.302 1", "far_chunk_00200"),
            ("15.304", "defines", "evaluation factors", "far_chunk_00300"),
            ("15.305", "references", "15.304", "far_chunk_00300"),
            ("contracting officer", "authorizes", "sole source", "far_chunk_00100"),
            ("part 15", "applies_to", "negotiation", "far_chunk_00350"),
        ],
    )
    con.execute(f"COPY triples TO '{kg_path}' (FORMAT PARQUET)")
    con.close()
    return kg_path


class TestFindRelated:
    def test_finds_by_subject(self, tmp_kg: Path) -> None:
        results = find_related("6.302-1", kg_path=tmp_kg)
        assert len(results) > 0
        assert any("6.302" in t.subject for t in results)

    def test_finds_by_object(self, tmp_kg: Path) -> None:
        results = find_related("sole source", kg_path=tmp_kg)
        assert len(results) > 0
        assert any(t.object == "sole source" for t in results)


class TestFindByRelation:
    def test_finds_references(self, tmp_kg: Path) -> None:
        results = find_by_relation("references", kg_path=tmp_kg)
        assert len(results) >= 2
        assert all(t.relation == "references" for t in results)


class TestMultiHop:
    def test_discovers_connected_entities(self, tmp_kg: Path) -> None:
        results = multi_hop("sole source", hops=2, kg_path=tmp_kg)
        assert len(results) > 0
        # Should discover at least 6.302-1 and contracting officer
        all_entities = {t.subject for t in results} | {t.object for t in results}
        assert "6.302-1" in all_entities or any("6.302" in e for e in all_entities)


class TestGetEntityNeighbors:
    def test_returns_grouped_neighbors(self, tmp_kg: Path) -> None:
        neighbors = get_entity_neighbors("6.302-1", kg_path=tmp_kg)
        assert isinstance(neighbors, dict)
        assert len(neighbors) > 0
        # Should have at least "authorizes" and "requires" relations
        all_values = [v for vals in neighbors.values() for v in vals]
        assert len(all_values) > 0


class TestProvenanceChunkIds:
    def test_extracts_unique_ids(self) -> None:
        triples = [
            Triple("a", "r", "b", "chunk_1"),
            Triple("c", "r", "d", "chunk_2"),
            Triple("e", "r", "f", "chunk_1"),
        ]
        ids = get_provenance_chunk_ids(triples)
        assert ids == ["chunk_1", "chunk_2"]


# --- Graph answer tests ---


class TestExtractQueryEntities:
    def test_extracts_section_references(self) -> None:
        entities = _extract_query_entities("What does FAR 6.302 cover?")
        assert any("6.302" in e for e in entities)

    def test_extracts_concept_words(self) -> None:
        entities = _extract_query_entities("Which sections relate to sole-source exceptions?")
        assert any("sole" in e for e in entities)


class TestGenerateGraphAnswer:
    def test_fallback_without_llm(self, tmp_kg: Path) -> None:
        result = generate_graph_answer("What relates to sole source contracting?", kg_path=tmp_kg)
        assert isinstance(result, GraphAnswer)
        assert len(result.triples_used) > 0
        assert len(result.provenance_chunk_ids) > 0
        assert "sole source" in result.answer.lower() or "triples" in result.answer.lower()

    def test_no_results_message(self, tmp_kg: Path) -> None:
        result = generate_graph_answer("Tell me about quantum computing", kg_path=tmp_kg)
        assert isinstance(result, GraphAnswer)
        # Either no triples or a "not found" message
        assert result.triples_used == [] or len(result.answer) > 0
