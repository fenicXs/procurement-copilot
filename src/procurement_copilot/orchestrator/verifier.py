"""Verifier: check that every non-trivial claim in an answer is grounded.

The verifier inspects the generated answer against the evidence (retrieved
chunks, SQL results, or graph triples) and flags unsupported claims.
When no LLM is available it uses simple heuristic overlap checks.
"""

import logging
import re
from dataclasses import dataclass, field

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


def _sentence_has_evidence(
    sentence: str, evidence_text: str, threshold: float = 0.4
) -> bool:
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


def verify_answer(
    answer: str,
    evidence_texts: list[str],
    abstain_on_failure: bool = True,
    threshold: float = 0.4,
) -> VerificationResult:
    """Verify that claims in an answer are supported by provided evidence.

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
