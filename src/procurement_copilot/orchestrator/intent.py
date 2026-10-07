"""Intent classification for routing user queries to the correct pipeline."""

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

# Intent categories
INTENT_RAG = "rag"
INTENT_SQL = "sql"
INTENT_GRAPH = "graph"
INTENT_MIXED = "mixed"

# SQL-signal patterns: questions about spending, amounts, counts, agencies, recipients
_SQL_SIGNALS = re.compile(
    r"\b(?:how much|how many|total|sum|average|avg|count|top \d+|largest|smallest|"
    r"spending|spent|awarded|obligated|outlays?|dollars?|amount|budget|"
    r"agency|agencies|recipient|contractor|vendor|contract type|"
    r"rank|compare|breakdown|between \d{4} and \d{4}|"
    r"fiscal year|fy\s?\d{2,4})\b",
    re.IGNORECASE,
)

# Graph-signal patterns: relationship queries, multi-hop, connections
_GRAPH_SIGNALS = re.compile(
    r"\b(?:relat(?:ed?|ionship|es? to)|connect(?:ed|ion)|linked? to|"
    r"cross[- ]reference|references?|defines?|authoriz(?:es?|ed)|"
    r"requires?|applies? to|exception to|delegates? to|"
    r"neighbors?|multi[- ]hop|graph|triples?|entities|"
    r"what (?:does|is) .+ (?:related|connected) to)\b",
    re.IGNORECASE,
)

# RAG-signal patterns: policy questions, compliance, what/how/when about FAR
_RAG_SIGNALS = re.compile(
    r"\b(?:FAR\s+\d|policy|regulation|compliance|requirement|clause|provision|"
    r"what (?:does|is|are)|explain|describe|summarize|"
    r"when (?:must|should|is)|who (?:must|should|is|can)|"
    r"approval|threshold|procedure|process|rule|guidance|"
    r"sole[- ]source|set[- ]aside|small business|competitive|"
    r"contracting officer|sealed bidding|negotiation)\b",
    re.IGNORECASE,
)


INTENT_SYSTEM_PROMPT = """You classify a user's question about federal procurement into \
exactly one category. Respond with exactly one word, no punctuation, no explanation:

- rag: policy/compliance questions about FAR regulations (what does FAR say about X,
  approval requirements, definitions, procedures)
- sql: questions about spending data — award amounts, counts, agencies, recipients
  (how much, how many, top N, total, average)
- graph: questions about relationships/connections between FAR concepts (how is X
  related to Y, what does X authorize, cross-references)
- mixed: questions that need both policy context AND spending data together

Respond with exactly one of: rag, sql, graph, mixed
"""


def classify_intent_llm(question: str, llm: "BaseChatModel") -> str | None:
    """LLM-based intent classification.

    Returns None (caller falls back to regex) if the response doesn't parse
    into one of the four known labels — plain-text parsing rather than
    structured output, since not every configured provider (e.g. the Bytez
    wrapper) supports tool-calling-based structured output.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [SystemMessage(content=INTENT_SYSTEM_PROMPT), HumanMessage(content=question)]
    response = llm.invoke(messages)
    content = (
        (response.content if isinstance(response.content, str) else str(response.content))
        .strip()
        .lower()
    )

    for label in (INTENT_RAG, INTENT_SQL, INTENT_GRAPH, INTENT_MIXED):
        if label in content:
            return label
    return None


def classify_intent_regex(question: str) -> str:
    """Classify a user question into one of: rag, sql, graph, mixed, via
    keyword heuristics. Offline fallback for classify_intent()."""
    sql_score = len(_SQL_SIGNALS.findall(question))
    graph_score = len(_GRAPH_SIGNALS.findall(question))
    rag_score = len(_RAG_SIGNALS.findall(question))

    logger.info(
        "Intent scores — rag: %d, sql: %d, graph: %d",
        rag_score,
        sql_score,
        graph_score,
    )

    total = sql_score + graph_score + rag_score

    if total == 0:
        # Default to RAG for unknown queries
        return INTENT_RAG

    # If two categories both score above threshold, call it mixed
    scores = {"rag": rag_score, "sql": sql_score, "graph": graph_score}
    top_two = sorted(scores.values(), reverse=True)
    if top_two[0] > 0 and top_two[1] > 0 and top_two[1] >= top_two[0] * 0.5:
        top_intents = [k for k, v in scores.items() if v == top_two[0] or v == top_two[1]]
        if len(set(top_intents)) >= 2:
            return INTENT_MIXED

    # Single dominant intent
    dominant = max(scores, key=lambda k: scores[k])
    return dominant


def classify_intent(question: str) -> str:
    """Classify a user question into one of: rag, sql, graph, mixed.

    Uses an LLM when a provider is configured (falls back to regex if the
    response doesn't parse); uses the regex heuristics directly offline.
    """
    from procurement_copilot.llm import get_llm

    llm = get_llm()
    if llm is not None:
        result = classify_intent_llm(question, llm)
        if result is not None:
            return result
        logger.warning("LLM intent classification unparseable — falling back to regex.")

    return classify_intent_regex(question)
