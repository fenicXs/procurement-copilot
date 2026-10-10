"""In-memory cache of fully verified answers, keyed by the normalized question.

Repeated questions (the demo chips, anything shown to a stakeholder) are
answered with zero LLM calls, which matters on free tiers: Groq's gpt-oss-20b
allows only 200k tokens/day (~30-40 RAG questions). Per-instance and lost on
scale-to-zero by design — it is a token saver, not a datastore.

Only stores answers that were verified, did not abstain, and had no error, so a
transient failure or a trimmed/unverified answer is never replayed.
"""

import copy
import re
import threading
import time
from collections import OrderedDict
from typing import Any

from procurement_copilot.config import settings

_lock = threading.Lock()
_entries: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()

_NON_WORD_RE = re.compile(r"[^a-z0-9.\-\s]")
_SPACE_RE = re.compile(r"\s+")


def normalize_question(question: str) -> str:
    """Case/punctuation/whitespace-insensitive key: 'Top 5 recipients?' == 'top 5 recipients'."""
    return _SPACE_RE.sub(" ", _NON_WORD_RE.sub(" ", question.lower())).strip()


def _enabled() -> bool:
    return settings.ANSWER_CACHE_TTL_SECONDS > 0 and settings.ANSWER_CACHE_MAX_ENTRIES > 0


def get(question: str) -> Any | None:
    """A copy of the cached response, or None on miss/expiry/disabled."""
    if not _enabled():
        return None
    key = normalize_question(question)
    now = time.monotonic()
    with _lock:
        item = _entries.get(key)
        if item is None:
            return None
        stored_at, response = item
        if now - stored_at > settings.ANSWER_CACHE_TTL_SECONDS:
            del _entries[key]
            return None
        _entries.move_to_end(key)  # LRU
        return copy.deepcopy(response)


def put(question: str, response: Any) -> None:
    if not _enabled():
        return
    key = normalize_question(question)
    if not key:
        return
    with _lock:
        _entries[key] = (time.monotonic(), copy.deepcopy(response))
        _entries.move_to_end(key)
        while len(_entries) > settings.ANSWER_CACHE_MAX_ENTRIES:
            _entries.popitem(last=False)


def clear() -> None:
    with _lock:
        _entries.clear()
