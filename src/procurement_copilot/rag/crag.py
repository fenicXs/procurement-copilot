"""Corrective-RAG (CRAG): grade retrieved documents, rewrite the query and
retry when they're weak, abstain instead of hallucinating when retries run
out — rather than always generating from whatever the first retrieval call
happened to return.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from procurement_copilot.rag.retriever import RetrievedChunk

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

GRADE_SUFFICIENT: Literal["sufficient"] = "sufficient"
GRADE_INSUFFICIENT: Literal["insufficient"] = "insufficient"
Grade = Literal["sufficient", "insufficient"]

GRADE_SYSTEM_PROMPT = """You are grading whether retrieved excerpts from the Federal \
Acquisition Regulation (FAR) contain enough information to answer a user's question.

Respond with exactly one word: "sufficient" or "insufficient".
- "sufficient": the excerpts address the question, even partially.
- "insufficient": the excerpts are off-topic or do not address the question at all.
"""

REWRITE_SYSTEM_PROMPT = """You rewrite search queries to improve retrieval against the \
Federal Acquisition Regulation (FAR). The previous query didn't retrieve useful results.
Rewrite it to be more specific and use FAR terminology. Respond with ONLY the rewritten
query — no explanation, no quotes.
"""


def grade_documents(query: str, chunks: list[RetrievedChunk], llm: "BaseChatModel | None") -> Grade:
    """LLM-as-judge: do these chunks contain enough to answer the query?

    No chunks is always insufficient. No LLM available (offline mode) always
    passes whatever was retrieved through — there's no judge to consult.
    """
    if not chunks:
        return GRADE_INSUFFICIENT
    if llm is None:
        return GRADE_SUFFICIENT

    from langchain_core.messages import HumanMessage, SystemMessage

    context = "\n\n".join(c.text[:500] for c in chunks)
    messages = [
        SystemMessage(content=GRADE_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {query}\n\nExcerpts:\n{context}"),
    ]
    response = llm.invoke(messages)
    verdict = (
        (response.content if isinstance(response.content, str) else str(response.content))
        .strip()
        .lower()
    )
    return GRADE_INSUFFICIENT if "insufficient" in verdict else GRADE_SUFFICIENT


def rewrite_query(query: str, llm: "BaseChatModel | None") -> str:
    """Rewrite the query to improve retrieval. Unchanged (no-op) offline."""
    if llm is None:
        return query

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    response = llm.invoke(messages)
    rewritten = (
        response.content if isinstance(response.content, str) else str(response.content)
    ).strip()
    return rewritten or query


def run_crag_loop(
    question: str,
    retrieve_fn: Callable[[str], list[RetrievedChunk]],
    llm: "BaseChatModel | None",
    max_retries: int = MAX_RETRIES,
) -> tuple[list[RetrievedChunk], str, bool]:
    """Retrieve → grade → rewrite/retry → (chunks, final query, abstained).

    Bounded to `max_retries` rewrites. `retrieve_fn` and `llm` are injected
    so this loop is testable without a real vector index or LLM call.
    """
    query = question
    chunks: list[RetrievedChunk] = []

    for attempt in range(max_retries + 1):
        chunks = retrieve_fn(query)
        grade = grade_documents(question, chunks, llm)

        if grade == GRADE_SUFFICIENT:
            return chunks, query, False

        if attempt < max_retries:
            query = rewrite_query(query, llm)
            logger.info("CRAG: insufficient docs (attempt %d) — rewrote query.", attempt + 1)

    logger.warning("CRAG: exhausted %d retries — abstaining.", max_retries)
    return chunks, query, True
