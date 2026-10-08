"""RAG answer generation: produce a grounded answer with citations from retrieved chunks."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from procurement_copilot.prompts import load_prompt
from procurement_copilot.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

ABSTAIN_MESSAGE = (
    "I don't have sufficiently grounded information in the FAR corpus to "
    "answer this question confidently."
)


@dataclass
class Citation:
    """A citation reference to a source chunk."""

    chunk_id: str
    page_start: int
    page_end: int
    section_heading: str
    source_url: str


@dataclass
class RAGAnswer:
    """Answer produced by the RAG pipeline."""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    abstained: bool = False


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        heading = f" ({chunk.section_heading})" if chunk.section_heading else ""
        parts.append(
            f"[{i}] {chunk.chunk_id} — pages {chunk.page_start}-{chunk.page_end}"
            f"{heading}:\n{chunk.text}"
        )
    return "\n\n".join(parts)


def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Build citation objects from retrieved chunks."""
    return [
        Citation(
            chunk_id=c.chunk_id,
            page_start=c.page_start,
            page_end=c.page_end,
            section_heading=c.section_heading,
            source_url=c.source_url,
        )
        for c in chunks
    ]


def _get_llm():  # type: ignore[no-untyped-def]
    """Get the best available chat model."""
    from procurement_copilot.llm import get_llm

    return get_llm()


RAG_SYSTEM_PROMPT = load_prompt("rag_answer")


def generate_rag_answer(
    question: str,
    chunks: list[RetrievedChunk],
) -> RAGAnswer:
    """Generate a grounded answer with citations from retrieved chunks.

    If no LLM is available, returns a structured summary of the retrieved chunks.
    """
    citations = _build_citations(chunks)
    context = _format_context(chunks)

    llm = _get_llm()

    if llm is None:
        # Fallback: return a structured summary without LLM
        logger.warning("No LLM available — returning chunk summary as answer.")
        summary_parts = []
        for c in chunks[:3]:
            heading = f" ({c.section_heading})" if c.section_heading else ""
            summary_parts.append(
                f"From {c.chunk_id} (pages {c.page_start}-{c.page_end}){heading}: "
                f"{c.text[:300]}..."
            )
        answer_text = f"Based on {len(chunks)} retrieved FAR sections:\n\n" + "\n\n".join(
            summary_parts
        )
        return RAGAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_chunks=chunks,
        )

    # LLM-based answer generation
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    response = llm.invoke(messages)
    answer_text = response.content if isinstance(response.content, str) else str(response.content)

    return RAGAnswer(
        question=question,
        answer=answer_text,
        citations=citations,
        retrieved_chunks=chunks,
    )


def generate_rag_answer_with_retrieval(
    question: str,
    top_k: int = 5,
    index_dir: Path | None = None,
) -> RAGAnswer:
    """Full Corrective-RAG pipeline: retrieve -> grade -> rewrite/retry -> generate.

    Replaces a plain retrieve()-then-generate_rag_answer() call: retrieval is
    graded by an LLM judge, the query is rewritten and retried (bounded) when
    the pool is weak, and the pipeline abstains rather than generating from
    an insufficient context when retries are exhausted.
    """
    from procurement_copilot.rag.crag import run_crag_loop
    from procurement_copilot.rag.retriever import retrieve

    llm = _get_llm()

    def _retrieve(q: str) -> list[RetrievedChunk]:
        return retrieve(q, top_k=top_k, index_dir=index_dir)

    chunks, _final_query, abstained = run_crag_loop(question, _retrieve, llm)

    if abstained:
        return RAGAnswer(
            question=question,
            answer=ABSTAIN_MESSAGE,
            citations=[],
            retrieved_chunks=chunks,
            abstained=True,
        )

    return generate_rag_answer(question, chunks)
