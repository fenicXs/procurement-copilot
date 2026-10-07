"""LangFuse tracing — optional, gated on API keys being configured.

Uses the current (v4, OTel-based) Langfuse SDK: `langfuse.langchain.CallbackHandler`.
Returns None when unset so offline/CI runs are unaffected; callers pass the
handler into LangChain/LangGraph `config={"callbacks": [handler]}`, which
propagates to every node and nested LLM call via LangChain's ambient
RunnableConfig context — no per-call-site wiring needed beyond the top level.
"""

import logging
import os
from typing import TYPE_CHECKING

from procurement_copilot.config import settings

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

# Cache one Langfuse client per public key so we don't re-register/reconnect
# on every call — the SDK's CallbackHandler() looks up the active client
# registered under its public_key rather than taking credentials directly.
_client_cache: dict[str, "Langfuse"] = {}


def _get_key(name: str) -> str:
    return os.environ.get(name, "") or getattr(settings, name, "")


def _get_client(public_key: str, secret_key: str, host: str) -> "Langfuse":
    if public_key not in _client_cache:
        from langfuse import Langfuse

        _client_cache[public_key] = Langfuse(
            public_key=public_key, secret_key=secret_key, host=host
        )
    return _client_cache[public_key]


def get_langfuse_handler(session_id: str | None = None) -> "CallbackHandler | None":
    """Build a LangFuse CallbackHandler bound to a fresh trace id, or None if
    keys are not configured. Attach `session_id` via LangChain's
    `langfuse_session_id` metadata convention at the call site, e.g.:

        handler = get_langfuse_handler(session_id=...)
        config = {"callbacks": [handler], "metadata": {"langfuse_session_id": session_id}}
    """
    public_key = _get_key("LANGFUSE_PUBLIC_KEY")
    secret_key = _get_key("LANGFUSE_SECRET_KEY")
    host = _get_key("LANGFUSE_HOST") or settings.LANGFUSE_HOST

    if not public_key or not secret_key:
        return None

    from langfuse.langchain import CallbackHandler
    from langfuse.types import TraceContext

    client = _get_client(public_key, secret_key, host)
    trace_id = client.create_trace_id()

    logger.info("LangFuse tracing enabled (host=%s).", host)
    handler = CallbackHandler(trace_context=TraceContext(trace_id=trace_id))
    handler.trace_id = trace_id  # type: ignore[attr-defined]  # stashed for flush_and_get_trace_url()
    return handler


def flush_and_get_trace_url(handler: "CallbackHandler") -> str | None:
    """Flush buffered spans and return the trace's dashboard URL.

    Short-lived script/request processes can exit before the SDK's
    background thread sends buffered events — flush synchronously before
    returning, rather than relying on process-exit flushing.
    """
    public_key = _get_key("LANGFUSE_PUBLIC_KEY")
    secret_key = _get_key("LANGFUSE_SECRET_KEY")
    host = _get_key("LANGFUSE_HOST") or settings.LANGFUSE_HOST
    client = _get_client(public_key, secret_key, host)
    client.flush()
    return client.get_trace_url(trace_id=getattr(handler, "trace_id", None))
