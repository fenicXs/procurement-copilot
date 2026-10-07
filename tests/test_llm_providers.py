"""Tests for the Ollama provider branch in llm.py (construction-only, no network calls)."""

from langchain_ollama import ChatOllama, OllamaEmbeddings

from procurement_copilot.config import settings
from procurement_copilot.llm import get_embeddings, get_llm

# NOTE: settings.OLLAMA_* fields are read from the environment once at process
# start (pydantic-settings), so monkeypatching them mid-test has no effect on
# `settings` — only LLM_PROVIDER/API keys are read fresh via _get_key(os.environ).
# These tests assert against the real `settings` defaults instead.


def test_get_llm_defaults_to_none_without_provider(monkeypatch):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "BYTEZ_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    assert get_llm() is None


def test_get_llm_ollama_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    llm = get_llm()

    assert isinstance(llm, ChatOllama)
    assert llm.base_url == settings.OLLAMA_BASE_URL
    assert llm.model == settings.OLLAMA_MODEL
    assert llm.num_ctx == settings.OLLAMA_NUM_CTX


def test_get_llm_ollama_override_wins_over_cloud_keys(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    llm = get_llm()

    assert isinstance(llm, ChatOllama)


def test_get_embeddings_defaults_to_fake_without_provider(monkeypatch):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "BYTEZ_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    embeddings = get_embeddings()
    assert embeddings.__class__.__name__ == "FakeEmbeddings"


def test_get_embeddings_ollama_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    embeddings = get_embeddings()

    assert isinstance(embeddings, OllamaEmbeddings)
    assert embeddings.base_url == settings.OLLAMA_BASE_URL
    assert embeddings.model == settings.OLLAMA_EMBED_MODEL
