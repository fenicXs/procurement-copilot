"""Verifier: check that every non-trivial claim in an answer is grounded.

The verifier inspects the generated answer against the evidence (retrieved
chunks, SQL results, or graph triples) and flags unsupported claims.
When no LLM is available it uses simple heuristic overlap checks.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

from procurement_copilot.prompts import load_prompt

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying an answer against evidence."""

    is_grounded: bool
    original_answer: str
    verified_answer: str
    unsupported_claims: list[str] = field(default_factory=list)
    evidence_summary: str = ""


def _extract_sentences(text: str) -> list[str]:
    """Split text into sentences (simple heuristic)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _sentence_has_evidence(sentence: str, evidence_text: str, threshold: float = 0.4) -> bool:
    """Check if a sentence is supported by evidence via keyword overlap.

    A sentence is considered grounded if at least *threshold* fraction of its
    significant words (length > 3, non-stopwords) appear in the evidence.
    """
    stopwords = {
        "that",
        "this",
        "with",
        "from",
        "have",
        "been",
        "were",
        "will",
        "would",
        "could",
        "should",
        "does",
        "also",
        "than",
        "then",
        "which",
        "where",
        "when",
        "what",
        "their",
        "there",
        "they",
        "about",
        "into",
        "more",
        "some",
        "such",
        "each",
        "only",
        "other",
        "based",
        "most",
        "very",
    }

    words = re.findall(r"\b[a-zA-Z]{4,}\b", sentence.lower())
    significant = [w for w in words if w not in stopwords]

    if not significant:
        return True  # Trivial sentence, no verification needed

    evidence_lower = evidence_text.lower()
    matched = sum(1 for w in significant if w in evidence_lower)
    ratio = matched / len(significant)

    return ratio >= threshold


def _get_llm():  # type: ignore[no-untyped-def]
    from procurement_copilot.llm import get_llm

    return get_llm()


VERIFY_SYSTEM_PROMPT = load_prompt("verify")


# Five ~1000-char chunks plus SQL/graph evidence fit comfortably; the old 4000-char
# cap silently cut off the last chunk or two, which could hold the cited fact.
MAX_EVIDENCE_CHARS = 12000
LLM_VERIFY_ATTEMPTS = 2

_GROUNDED_RE = re.compile(r"GROUNDED:\s*(yes|no)", re.IGNORECASE)


def _judge_once(answer: str, evidence_block: str, llm: "BaseChatModel") -> tuple[bool | None, str]:
    """One judge call. Returns (verdict, raw_reply); verdict is None when the
    call failed or the reply had no parseable `GROUNDED: yes|no` line."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=VERIFY_SYSTEM_PROMPT),
        HumanMessage(content=f"Evidence:\n{evidence_block}\n\nAnswer:\n{answer}"),
    ]
    try:
        response = llm.invoke(messages)
    except Exception as exc:  # e.g. a provider 429/5xx — must not read as "ungrounded"
        logger.warning("Verifier LLM call failed: %s", exc)
        return None, ""
    content = response.content if isinstance(response.content, str) else str(response.content)
    match = _GROUNDED_RE.search(content)
    if match is None:
        return None, content
    return match.group(1).lower() == "yes", content


def _verify_answer_llm(
    answer: str,
    evidence_texts: list[str],
    llm: "BaseChatModel",
    abstain_on_failure: bool = True,
) -> VerificationResult | None:
    """LLM-as-judge groundedness check — more accurate than keyword overlap,
    but requires a configured LLM provider.

    Returns None when the judge never produced a usable verdict (call error or
    unparseable reply on every attempt) so the caller can fall back to the
    keyword check instead of mistaking judge failure for an ungrounded answer.
    """
    evidence_block = "\n".join(evidence_texts)[:MAX_EVIDENCE_CHARS]

    verdict: bool | None = None
    content = ""
    for attempt in range(1, LLM_VERIFY_ATTEMPTS + 1):
        verdict, content = _judge_once(answer, evidence_block, llm)
        if verdict is not None:
            break
        logger.warning(
            "Verifier gave no usable verdict (attempt %d/%d). Reply: %.300r",
            attempt,
            LLM_VERIFY_ATTEMPTS,
            content,
        )

    if verdict is None:
        return None

    is_grounded = verdict
    if not is_grounded:
        # Surfaced at WARNING so rejections are visible in deployed logs.
        logger.warning("Verifier rejected answer. Judge reply: %.500r", content)
    unsupported_match = re.search(r"UNSUPPORTED:\s*(.*)", content, re.IGNORECASE)
    unsupported_raw = unsupported_match.group(1).strip() if unsupported_match else ""
    unsupported = (
        []
        if not unsupported_raw or unsupported_raw.upper().startswith("NONE")
        else [c.strip() for c in unsupported_raw.split(",") if c.strip()]
    )

    if is_grounded:
        verified_answer = answer
    elif abstain_on_failure:
        verified_answer = (
            "I cannot provide a verified answer to this question. "
            "The available evidence does not sufficiently support a response."
        )
    else:
        verified_answer = answer

    return VerificationResult(
        is_grounded=is_grounded,
        original_answer=answer,
        verified_answer=verified_answer,
        unsupported_claims=unsupported,
        evidence_summary=f"LLM-judged against {len(evidence_texts)} evidence sources.",
    )


def verify_answer(
    answer: str,
    evidence_texts: list[str],
    abstain_on_failure: bool = True,
    threshold: float = 0.4,
) -> VerificationResult:
    """Verify that claims in an answer are supported by provided evidence.

    Uses an LLM-as-judge when a provider is configured (more accurate than
    keyword overlap); falls back to the keyword-overlap heuristic offline.

    Args:
        answer: The generated answer text.
        evidence_texts: List of evidence strings (chunk texts, SQL result
            summaries, triple descriptions) to check against.
        abstain_on_failure: If True, replace unsupported claims with a
            disclaimer. If False, keep the original answer with warnings.

    Returns:
        VerificationResult with grounded status and possibly revised answer.
    """
    if not answer or not evidence_texts:
        return VerificationResult(
            is_grounded=not answer,
            original_answer=answer,
            verified_answer=answer or "No answer was generated.",
            unsupported_claims=[],
            evidence_summary="No evidence provided.",
        )

    llm = _get_llm()
    if llm is not None:
        llm_result = _verify_answer_llm(
            answer, evidence_texts, llm, abstain_on_failure=abstain_on_failure
        )
        if llm_result is not None:
            return llm_result
        logger.warning("LLM verifier unusable — falling back to keyword-overlap check.")

    combined_evidence = "\n".join(evidence_texts)
    sentences = _extract_sentences(answer)
    unsupported: list[str] = []
    supported_sentences: list[str] = []

    for sentence in sentences:
        if _sentence_has_evidence(sentence, combined_evidence, threshold=threshold):
            supported_sentences.append(sentence)
        else:
            unsupported.append(sentence)

    is_grounded = len(unsupported) == 0

    if is_grounded:
        verified_answer = answer
    elif abstain_on_failure and len(unsupported) >= len(sentences):
        # All claims unsupported — abstain entirely
        verified_answer = (
            "I cannot provide a verified answer to this question. "
            "The available evidence does not sufficiently support a response."
        )
    elif abstain_on_failure and unsupported:
        # Partial support — keep supported claims, note the gap
        verified_answer = " ".join(supported_sentences)
        if verified_answer:
            verified_answer += (
                "\n\nNote: Some claims could not be verified against the "
                "available evidence and have been removed."
            )
        else:
            verified_answer = (
                "I cannot provide a verified answer to this question. "
                "The available evidence does not sufficiently support a response."
            )
    else:
        verified_answer = answer

    return VerificationResult(
        is_grounded=is_grounded,
        original_answer=answer,
        verified_answer=verified_answer,
        unsupported_claims=unsupported,
        evidence_summary=f"Checked against {len(evidence_texts)} evidence sources.",
    )
