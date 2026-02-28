"""Knowledge graph query engine: load triples and perform traversals."""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb

from procurement_copilot.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Triple:
    """A single (subject, relation, object) triple with provenance."""

    subject: str
    relation: str
    object: str
    chunk_id: str


def _load_kg(kg_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Load KG parquet into an in-memory DuckDB connection."""
    path = kg_path or settings.KG_PATH
    con = duckdb.connect()
    con.execute(f"CREATE TABLE triples AS SELECT * FROM read_parquet('{path}')")
    return con


def find_related(
    entity: str,
    kg_path: Path | None = None,
    max_results: int = 50,
) -> list[Triple]:
    """Find all triples where *entity* appears as subject or object."""
    con = _load_kg(kg_path)
    entity_norm = entity.strip().lower().replace("-", " ")

    rows = con.execute(
        "SELECT subject, relation, object, chunk_id FROM triples "
        "WHERE subject LIKE ? OR object LIKE ? "
        "LIMIT ?",
        [f"%{entity_norm}%", f"%{entity_norm}%", max_results],
    ).fetchall()
    con.close()

    return [Triple(subject=r[0], relation=r[1], object=r[2], chunk_id=r[3]) for r in rows]


def find_by_relation(
    relation: str,
    kg_path: Path | None = None,
    max_results: int = 50,
) -> list[Triple]:
    """Find all triples with a specific relation type."""
    con = _load_kg(kg_path)

    rows = con.execute(
        "SELECT subject, relation, object, chunk_id FROM triples " "WHERE relation = ? LIMIT ?",
        [relation.lower(), max_results],
    ).fetchall()
    con.close()

    return [Triple(subject=r[0], relation=r[1], object=r[2], chunk_id=r[3]) for r in rows]


def multi_hop(
    start_entity: str,
    hops: int = 2,
    kg_path: Path | None = None,
    max_per_hop: int = 20,
) -> list[Triple]:
    """Perform multi-hop traversal starting from *start_entity*.

    Returns all triples discovered within *hops* steps.
    """
    con = _load_kg(kg_path)
    start_norm = start_entity.strip().lower().replace("-", " ")

    discovered: list[Triple] = []
    seen_keys: set[tuple[str, str, str]] = set()
    frontier: set[str] = {start_norm}

    for hop in range(hops):
        if not frontier:
            break

        next_frontier: set[str] = set()
        for entity in frontier:
            rows = con.execute(
                "SELECT subject, relation, object, chunk_id FROM triples "
                "WHERE subject LIKE ? OR object LIKE ? "
                "LIMIT ?",
                [f"%{entity}%", f"%{entity}%", max_per_hop],
            ).fetchall()

            for r in rows:
                key = (r[0], r[1], r[2])
                if key not in seen_keys:
                    seen_keys.add(key)
                    discovered.append(
                        Triple(subject=r[0], relation=r[1], object=r[2], chunk_id=r[3])
                    )
                    # Add the "other end" to next frontier
                    if entity in r[0]:
                        next_frontier.add(r[2])
                    else:
                        next_frontier.add(r[0])

        frontier = next_frontier - {start_norm}

    con.close()
    return discovered


def get_entity_neighbors(
    entity: str,
    kg_path: Path | None = None,
) -> dict[str, list[str]]:
    """Get a map of {relation: [connected_entities]} for an entity."""
    triples = find_related(entity, kg_path)
    entity_norm = entity.strip().lower().replace("-", " ")

    neighbors: dict[str, list[str]] = {}
    for t in triples:
        other = t.object if entity_norm in t.subject else t.subject
        neighbors.setdefault(t.relation, []).append(other)

    # Deduplicate lists
    return {k: sorted(set(v)) for k, v in neighbors.items()}


def get_provenance_chunk_ids(triples: list[Triple]) -> list[str]:
    """Extract unique chunk_ids from a list of triples for citation."""
    return sorted({t.chunk_id for t in triples})
