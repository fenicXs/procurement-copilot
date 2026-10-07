"""Tests for the LangFuse handler factory (construction-only, no network calls).

NOTE: `settings` is a module-level singleton loaded from `.env` at import
time, so deleting an env var alone doesn't erase an already-loaded real key —
tests that simulate "no keys configured" must also monkeypatch the `settings`
object itself.
"""

from langfuse.langchain import CallbackHandler

from procurement_copilot.config import settings
from procurement_copilot.observability import get_langfuse_handler


def test_returns_none_without_keys(monkeypatch):
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "")
    assert get_langfuse_handler() is None


def test_returns_handler_with_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-dummy")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-dummy")

    handler = get_langfuse_handler(session_id="test-session")

    assert isinstance(handler, CallbackHandler)


def test_missing_secret_key_still_returns_none(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-dummy")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "")
    assert get_langfuse_handler() is None
