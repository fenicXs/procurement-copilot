"""Extract entity-relation triples from FAR chunks and store as Parquet.

Usage:
    python scripts/build_kg.py [--input data/processed/far_chunks.jsonl]
                               [--output data/processed/kg.parquet]

Uses rule-based extraction to identify FAR entities (sections, concepts,
thresholds, roles) and their relationships. Each triple includes provenance
(source chunk_id) for citation support.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "far_chunks.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "kg.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Rule-based entity/relation extraction patterns ---

# FAR section references like "FAR 6.302", "Part 15", "Subpart 19.5", "52.212-4"
_SECTION_RE = re.compile(
    r"\b(?:FAR\s+)?(\d{1,2}\.\d{2,4}(?:-\d+)?)\b"
    r"|\b(Part\s+\d+(?:\.\d+)?)\b"
    r"|\b(Subpart\s+\d+\.\d+)\b",
    re.IGNORECASE,
)

# Key procurement concepts
_CONCEPTS = [
    "sole source",
    "sole-source",
    "small business",
    "competition",
    "competitive",
    "simplified acquisition",
    "micro-purchase",
    "sealed bidding",
    "negotiation",
    "cost accounting",
    "cost reimbursement",
    "fixed-price",
    "fixed price",
    "indefinite delivery",
    "indefinite quantity",
    "blanket purchase",
    "task order",
    "delivery order",
    "protest",
    "debarment",
    "suspension",
    "responsibility",
    "responsiveness",
    "best value",
    "lowest price",
    "technical evaluation",
    "evaluation factor",
    "evaluation factors",
    "evaluation criteria",
    "past performance",
    "organizational conflict",
    "subcontracting",
    "commercial item",
    "commercial product",
    "commercial service",
    "justification and approval",
    "market research",
    "source selection",
    "option",
    "modification",
    "termination",
    "default",
    "convenience",
    "wage determination",
    "service contract",
    "construction contract",
    "architect-engineer",
    "Davis-Bacon",
    "Buy American",
    "trade agreement",
    "foreign acquisition",
    "environmental",
    "sustainability",
    "cybersecurity",
    "intellectual property",
    "data rights",
    "warranties",
    "inspection",
    "acceptance",
    "payment",
    "invoice",
    "financing",
    "bonds",
    "insurance",
    "taxes",
    "contingent fees",
    "gratuities",
    "kickbacks",
    "whistleblower",
    "ethics",
    "disclosure",
    "certified cost",
    "Truth in Negotiations",
    "TINA",
]

_CONCEPT_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _CONCEPTS) + r")\b",
    re.IGNORECASE,
)

# Roles/actors
_ROLES = [
    "contracting officer",
    "contractor",
    "offeror",
    "agency head",
    "small business administration",
    "SBA",
    "GAO",
    "inspector general",
    "head of contracting activity",
    "HCA",
    "competition advocate",
    "procurement executive",
    "chief acquisition officer",
]

_ROLE_RE = re.compile(
    r"\b(" + "|".join(re.escape(r) for r in _ROLES) + r")\b",
    re.IGNORECASE,
)

# Relationship signal phrases
_RELATION_PATTERNS: list[tuple[str, re.Pattern]] = [  # type: ignore[type-arg]
    ("requires", re.compile(r"\brequires?\b|\bmust\b|\bshall\b", re.IGNORECASE)),
    ("applies_to", re.compile(r"\bappl(?:y|ies)\s+to\b", re.IGNORECASE)),
    ("exception_to", re.compile(r"\bexcept(?:ion|ed)?\b|\bexempt\b|\bwaiver?\b", re.IGNORECASE)),
    ("references", re.compile(r"\bsee\s+(?:also\s+)?(?:FAR\s+)?\d", re.IGNORECASE)),
    ("defines", re.compile(r"\bmeans?\b|\bdefined?\s+(?:as|in)\b", re.IGNORECASE)),
    ("authorizes", re.compile(r"\bauthoriz\w+\b|\bpermit\w*\b|\ballow\w*\b", re.IGNORECASE)),
    ("prohibits", re.compile(r"\bprohibit\w*\b|\bforbid\w*\b|\bnot\s+permit\b", re.IGNORECASE)),
    ("delegates_to", re.compile(r"\bdelegate\w*\b|\bdesignate\w*\b", re.IGNORECASE)),
    ("threshold", re.compile(r"\$[\d,]+(?:\.\d{2})?\b|\bthreshold\b", re.IGNORECASE)),
]


def _normalize(entity: str) -> str:
    """Normalize an entity string."""
    return entity.strip().lower().replace("-", " ").replace("  ", " ")


def extract_triples(chunk_id: str, text: str) -> list[dict]:
    """Extract (subject, relation, object, chunk_id) triples from text."""
    triples: list[dict] = []

    # Find all entities in this chunk
    sections = [m.group(0) for m in _SECTION_RE.finditer(text)]
    concepts = list({m.group(0) for m in _CONCEPT_RE.finditer(text)})
    roles = list({m.group(0) for m in _ROLE_RE.finditer(text)})

    # Detect which relations are present
    active_relations = []
    for rel_name, pattern in _RELATION_PATTERNS:
        if pattern.search(text):
            active_relations.append(rel_name)

    if not active_relations:
        active_relations = ["mentions"]

    # Generate triples: section <-> concept, section <-> role, concept <-> concept
    for section in sections:
        for concept in concepts:
            for rel in active_relations[:2]:  # Limit to top 2 relations per pair
                triples.append(
                    {
                        "subject": _normalize(section),
                        "relation": rel,
                        "object": _normalize(concept),
                        "chunk_id": chunk_id,
                    }
                )
        for role in roles:
            for rel in active_relations[:1]:
                triples.append(
                    {
                        "subject": _normalize(section),
                        "relation": rel,
                        "object": _normalize(role),
                        "chunk_id": chunk_id,
                    }
                )

    # Section cross-references
    if len(sections) > 1:
        for i, s1 in enumerate(sections):
            for s2 in sections[i + 1 :]:
                triples.append(
                    {
                        "subject": _normalize(s1),
                        "relation": "references",
                        "object": _normalize(s2),
                        "chunk_id": chunk_id,
                    }
                )

    # Role-concept relationships
    for role in roles:
        for concept in concepts[:3]:  # Limit explosion
            for rel in active_relations[:1]:
                triples.append(
                    {
                        "subject": _normalize(role),
                        "relation": rel,
                        "object": _normalize(concept),
                        "chunk_id": chunk_id,
                    }
                )

    return triples


def build_kg(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Build knowledge graph triples from FAR chunks and save as Parquet."""
    import duckdb

    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    logger.info("Processing %d chunks for triple extraction...", len(chunks))

    all_triples: list[dict] = []
    for chunk in chunks:
        triples = extract_triples(chunk["chunk_id"], chunk["text"])
        all_triples.extend(triples)

    logger.info("Extracted %d raw triples.", len(all_triples))

    # Deduplicate
    seen: set[tuple[str, str, str]] = set()
    unique_triples: list[dict] = []
    for t in all_triples:
        key = (t["subject"], t["relation"], t["object"])
        if key not in seen:
            seen.add(key)
            unique_triples.append(t)

    logger.info("Deduplicated to %d unique triples.", len(unique_triples))

    # Save as Parquet via DuckDB
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE triples (subject VARCHAR, relation VARCHAR, "
        "object VARCHAR, chunk_id VARCHAR)"
    )
    if unique_triples:
        con.executemany(
            "INSERT INTO triples VALUES (?, ?, ?, ?)",
            [(t["subject"], t["relation"], t["object"], t["chunk_id"]) for t in unique_triples],
        )
    con.execute(f"COPY triples TO '{output_path}' (FORMAT PARQUET)")
    con.close()

    logger.info("Saved KG to %s.", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build knowledge graph from FAR chunks")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_kg(args.input, args.output)


if __name__ == "__main__":
    main()
    sys.exit(0)
