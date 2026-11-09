"""Graph-based answer generation with provenance citations."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from procurement_copilot.graph.query import (
    Triple,
    find_by_relation,
    find_related,
    get_provenance_chunk_ids,
    multi_hop,
)
from procurement_copilot.prompts import load_prompt

logger = logging.getLogger(__name__)


@dataclass
class GraphAnswer:
    """Answer produced by the GraphRAG pipeline."""

    question: str
    answer: str
    triples_used: list[Triple] = field(default_factory=list)
    provenance_chunk_ids: list[str] = field(default_factory=list)


def _extract_query_entities(question: str) -> list[str]:
    """Extract potential entity names from the question using simple heuristics.

    Prioritizes multi-word procurement phrases before falling back to
    single-word extraction.
    """
    import re

    entities: list[str] = []

    # FAR section references (e.g., "6.302-1", "FAR 15.304")
    section_matches = re.findall(
        r"\b(?:FAR\s+)?(\d{1,2}\.\d{2,4}(?:-\d+)?)\b", question, re.IGNORECASE
    )
    entities.extend(section_matches)

    # Known multi-word procurement phrases (checked first)
    known_phrases = [
        "contracting officer",
        "sole source",
        "small business",
        "full and open competition",
        "best value",
        "past performance",
        "cost accounting",
        "market research",
        "source selection",
        "sealed bidding",
        "contract type",
        "fixed price",
        "cost reimbursement",
        "commercial items",
        "commercial products",
        "micro purchase",
        "simplified acquisition",
        "subcontracting plan",
        "set aside",
        "set-aside",
        "evaluation factors",
        "responsible contractor",
    ]
    q_lower = question.lower()
    matched_words: set[str] = set()
    for phrase in known_phrases:
        if phrase in q_lower:
            entities.append(phrase)
            matched_words.update(phrase.split())

    # Single-word fallback for unmatched terms
    stopwords = {
        "what",
        "which",
        "where",
        "when",
        "does",
        "that",
        "this",
        "have",
        "with",
        "from",
        "about",
        "related",
        "relate",
        "sections",
        "section",
        "requirements",
        "rules",
        "regulations",
        "show",
        "graph",
        "triples",
        "connected",
        "knowledge",
        "entities",
        "linked",
        "connections",
        "explain",
        "list",
    }
    words = re.findall(r"\b[a-zA-Z-]{4,}\b", q_lower)
    for w in words:
        if w not in stopwords and w not in matched_words:
            entities.append(w)

    return entities[:5]


# Words in a question that name a relation type in the KG (relation column values).
_RELATION_PATTERNS: list[tuple[str, str]] = [
    (r"\bauthori[sz]\w*", "authorizes"),
    (r"\b(?:cross[- ]?)?refer\w*", "references"),
    (r"\brequire\w*", "requires"),
    (r"\bmention\w*", "mentions"),
    (r"\bexception\w*", "exception_to"),
    (r"\bdefin(?:e|es|ed|ition)\b", "defines"),
    (r"\bprohibit\w*", "prohibits"),
    (r"\bapplies? to\b|\bapplicab\w*", "applies_to"),
    (r"\bdelegat\w*", "delegates_to"),
]


def _extract_query_relations(question: str) -> list[str]:
    """Relation types the question names (e.g. 'authorizes', 'references'), in order."""
    import re

    q = question.lower()
    found: list[str] = []
    for pattern, relation in _RELATION_PATTERNS:
        if re.search(pattern, q) and relation not in found:
            found.append(relation)
    return found


def _get_llm():  # type: ignore[no-untyped-def]
    """Get the best available chat model."""
    from procurement_copilot.llm import get_llm

    return get_llm()


GRAPH_SYSTEM_PROMPT = load_prompt("graph_answer")

EMPTY_ANSWER_ATTEMPTS = 2  # one retry when the LLM returns an empty message


def _structured_summary(all_triples: list[Triple], chunk_ids: list[str]) -> str:
    """Deterministic, LLM-free summary of the triples, grouped by relation."""
    by_relation: dict[str, list[Triple]] = {}
    for t in all_triples:
        by_relation.setdefault(t.relation, []).append(t)

    parts = [f"Found {len(all_triples)} related triples across {len(chunk_ids)} chunks:\n"]
    for rel, triples in sorted(by_relation.items()):
        parts.append(f"\n{rel.upper()}:")
        for t in triples[:5]:
            parts.append(f"  - {t.subject} -> {t.object} [{t.chunk_id}]")
        if len(triples) > 5:
            parts.append(f"  ... and {len(triples) - 5} more")
    return "\n".join(parts)


def generate_graph_answer(
    question: str,
    kg_path: Path | None = None,
) -> GraphAnswer:
    """Generate an answer using knowledge graph traversal.

    Extracts entities from the question, queries the KG, and synthesizes
    an answer (with LLM if available, or structured summary fallback).
    """
    entities = _extract_query_entities(question)
    logger.info("Extracted entities from question: %s", entities)

    # Gather triples from multiple entity lookups + multi-hop
    all_triples: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()

    # When the question names a relation type ("authorizes", "references", ...), gather
    # triples of that type first: scoped to each entity, then a sample of the relation itself
    # (so questions with no usable entity, or whose entity matches nothing, still get evidence).
    relations = _extract_query_relations(question)
    for relation in relations:
        scoped = [
            t for entity in entities for t in find_related(entity, kg_path, 10, relation=relation)
        ]
        # An unscoped sample of the relation is only a fallback: when entity-scoped triples
        # exist, extra unrelated edges of that type would just dilute the evidence.
        extra = [] if scoped else find_by_relation(relation, kg_path, max_results=15)
        for t in scoped + extra:
            key = (t.subject, t.relation, t.object)
            if key not in seen:
                seen.add(key)
                all_triples.append(t)

    for entity in entities:
        # Direct lookup
        related = find_related(entity, kg_path, max_results=15)
        for t in related:
            key = (t.subject, t.relation, t.object)
            if key not in seen:
                seen.add(key)
                all_triples.append(t)

        # Multi-hop for the first 2 entities
        if len(entities) <= 2 or entities.index(entity) < 2:
            hops = multi_hop(entity, hops=2, kg_path=kg_path, max_per_hop=10)
            for t in hops:
                key = (t.subject, t.relation, t.object)
                if key not in seen:
                    seen.add(key)
                    all_triples.append(t)

    # Limit to most relevant triples
    all_triples = all_triples[:50]
    chunk_ids = get_provenance_chunk_ids(all_triples)

    if not all_triples:
        return GraphAnswer(
            question=question,
            answer="No relevant knowledge graph triples found for this question.",
            triples_used=[],
            provenance_chunk_ids=[],
        )

    # Format triples for display/LLM
    triple_lines = [
        f"  ({t.subject}) --[{t.relation}]--> ({t.object})  [{t.chunk_id}]" for t in all_triples
    ]
    triples_text = "\n".join(triple_lines)

    llm = _get_llm()

    if llm is None:
        logger.warning("No LLM available — returning triple summary.")
        return GraphAnswer(
            question=question,
            answer=_structured_summary(all_triples, chunk_ids),
            triples_used=all_triples,
            provenance_chunk_ids=chunk_ids,
        )

    # LLM-based answer
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=GRAPH_SYSTEM_PROMPT),
        HumanMessage(content=f"Knowledge graph triples:\n{triples_text}\n\nQuestion: {question}"),
    ]

    # Reasoning models (e.g. Groq gpt-oss) occasionally return an empty message with no error,
    # which neither the provider fallback nor the caller treats as a failure. Retry once, then
    # fall back to the structured summary rather than reporting "no answer".
    answer_text = ""
    for _attempt in range(EMPTY_ANSWER_ATTEMPTS):
        response = llm.invoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        answer_text = content.strip()
        if answer_text:
            break
        logger.warning("Graph LLM returned an empty answer (attempt %d).", _attempt + 1)
    if not answer_text:
        answer_text = _structured_summary(all_triples, chunk_ids)

    return GraphAnswer(
        question=question,
        answer=answer_text,
        triples_used=all_triples,
        provenance_chunk_ids=chunk_ids,
    )
