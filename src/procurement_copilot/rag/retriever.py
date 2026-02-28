"""RAG retriever: load FAISS index and retrieve relevant chunks for a query."""

import logging
from dataclasses import dataclass
from pathlib import Path

from procurement_copilot.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with metadata."""

    chunk_id: str
    text: str
    page_start: int
    page_end: int
    section_heading: str
    source_url: str
    score: float


def _get_embeddings():  # type: ignore[no-untyped-def]
    """Get the best available embedding model (must match what was used to build index)."""
    from procurement_copilot.llm import get_embeddings

    return get_embeddings()


def load_vectorstore(index_dir: Path | None = None):  # type: ignore[no-untyped-def]
    """Load the FAISS vectorstore from disk."""
    from langchain_community.vectorstores import FAISS

    path = index_dir or settings.VECTOR_INDEX_DIR
    embeddings = _get_embeddings()
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)


def retrieve(
    query: str,
    top_k: int = 5,
    index_dir: Path | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the top-k most relevant chunks for a query.

    Includes a simple reranking heuristic: boost chunks whose section heading
    contains words from the query.
    """
    vectorstore = load_vectorstore(index_dir)
    results = vectorstore.similarity_search_with_score(query, k=top_k * 2)

    chunks: list[RetrievedChunk] = []
    for doc, score in results:
        meta = doc.metadata
        chunks.append(
            RetrievedChunk(
                chunk_id=meta.get("chunk_id", ""),
                text=doc.page_content,
                page_start=meta.get("page_start", 0),
                page_end=meta.get("page_end", 0),
                section_heading=meta.get("section_heading", ""),
                source_url=meta.get("source_url", ""),
                score=float(score),
            )
        )

    # Simple reranking: boost chunks whose section heading overlaps query words
    query_words = {w.lower() for w in query.split() if len(w) > 3}
    for chunk in chunks:
        heading_words = {w.lower() for w in chunk.section_heading.split()}
        overlap = len(query_words & heading_words)
        if overlap:
            # Lower score = better for FAISS L2 distance
            chunk.score *= max(0.5, 1.0 - 0.1 * overlap)

    # Sort by score ascending (lower = more similar for L2)
    chunks.sort(key=lambda c: c.score)

    return chunks[:top_k]
