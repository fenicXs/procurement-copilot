"""Input guardrails for the public-facing query endpoint.

Lightweight, dependency-free checks run before a question reaches the
orchestrator: reject obvious PII, prompt-injection attempts, and junk input.
Not a replacement for a real moderation stack — scoped to what a small
compliance-domain demo plausibly needs to show it isn't wide open.
"""

import re

MAX_QUESTION_LENGTH = 1000
MIN_QUESTION_LENGTH = 3

# SSN-shaped (123-45-6789) or EIN-shaped (12-3456789) strings.
_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b\d{2}-\d{7}\b"),  # EIN
]

_INJECTION_PATTERNS = [
    re.compile(r"ignore (?:all )?(?:previous|above|prior) instructions", re.IGNORECASE),
    re.compile(r"disregard (?:all )?(?:previous|above|prior)", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"system prompt", re.IGNORECASE),
]


def check_input(question: str) -> tuple[bool, str | None]:
    """Return (ok, reason). `reason` is None when ok is True."""
    stripped = question.strip()

    if len(stripped) < MIN_QUESTION_LENGTH:
        return False, "Question is too short."
    if len(stripped) > MAX_QUESTION_LENGTH:
        return False, f"Question exceeds {MAX_QUESTION_LENGTH} characters."

    for pattern in _PII_PATTERNS:
        if pattern.search(stripped):
            return False, (
                "Question appears to contain personally identifiable "
                "information (SSN/EIN-shaped input)."
            )

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(stripped):
            return False, "Question contains a disallowed instruction-override pattern."

    return True, None
