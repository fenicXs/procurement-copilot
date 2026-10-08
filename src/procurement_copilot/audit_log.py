"""Append-only audit log of queries processed — timestamp, intent, grounding
outcome, and evidence provenance. A compliance-domain tool needs a record of
what was asked and what was used to answer it, not just the answer itself.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from procurement_copilot.config import settings

logger = logging.getLogger(__name__)


def log_query(
    question: str,
    intent: str,
    is_verified: bool,
    abstained: bool,
    citation_chunk_ids: list[str],
    session_id: str | None = None,
    log_path: Path | None = None,
) -> None:
    """Append one audit record. Never raises — a logging failure must not
    break the actual query response.
    """
    path = log_path or settings.AUDIT_LOG_PATH
    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "question": question,
        "intent": intent,
        "is_verified": is_verified,
        "abstained": abstained,
        "citation_chunk_ids": citation_chunk_ids,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        logger.exception("Failed to write audit log entry.")
