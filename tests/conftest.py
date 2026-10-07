"""Shared test fixtures.

`settings` is a module-level singleton loaded from `.env` at import time, so
a real local `.env` (API keys, LangFuse credentials, etc.) would otherwise
leak into every test — making the suite non-hermetic and, in LangFuse's
case, firing real network calls on every run. Clear all provider/tracing
config by default; individual tests opt back in via monkeypatch.
"""

import pytest

from procurement_copilot.config import settings

_PROVIDER_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "BYTEZ_API_KEY",
    "LLM_PROVIDER",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)


@pytest.fixture(autouse=True)
def _no_ambient_provider_config(monkeypatch):
    for key in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(settings, key, "", raising=False)
