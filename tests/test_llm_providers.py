"""Tests for the Ollama/Groq/fastembed provider branches in llm.py
(construction-only, no network calls)."""

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings

from procurement_copilot.config import settings
from procurement_copilot.llm import FastEmbedEmbeddings, get_embeddings, get_llm

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


def test_get_llm_groq_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")

    llm = get_llm()

    assert isinstance(llm, ChatGroq)
    assert llm.model_name == settings.GROQ_MODEL


def test_get_llm_gemini_override(monkeypatch):
    from langchain_google_genai import ChatGoogleGenerativeAI

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    llm = get_llm()

    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model.endswith(settings.GEMINI_MODEL)


def test_get_llm_provider_override_arg_wins_over_llm_provider(monkeypatch):
    from langchain_google_genai import ChatGoogleGenerativeAI

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    assert isinstance(get_llm("gemini"), ChatGoogleGenerativeAI)
    assert isinstance(get_llm(), ChatGroq)  # no override -> unchanged behaviour


def test_verifier_uses_separate_provider_when_configured(monkeypatch):
    from langchain_google_genai import ChatGoogleGenerativeAI

    from procurement_copilot.orchestrator import verifier

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    assert isinstance(verifier._get_llm(), ChatGroq)  # default: same as generator

    monkeypatch.setenv("VERIFIER_LLM_PROVIDER", "gemini")
    assert isinstance(verifier._get_llm(), ChatGoogleGenerativeAI)


def test_verifier_falls_back_to_default_provider_when_judge_provider_breaks(monkeypatch):
    from procurement_copilot import llm as llm_module
    from procurement_copilot.orchestrator import verifier

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")
    monkeypatch.setenv("VERIFIER_LLM_PROVIDER", "gemini")

    def _boom():
        raise ValueError("missing GEMINI_API_KEY")

    monkeypatch.setattr(llm_module, "_gemini_chat", _boom)

    assert isinstance(verifier._get_llm(), ChatGroq)


def test_get_llm_groq_override_wins_over_cloud_keys(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    llm = get_llm()

    assert isinstance(llm, ChatGroq)


def test_get_embeddings_fastembed_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")

    embeddings = get_embeddings()

    assert isinstance(embeddings, FastEmbedEmbeddings)
    assert embeddings.model_name == settings.FASTEMBED_DENSE_MODEL


def test_get_embeddings_embedding_provider_independent_of_llm_provider(monkeypatch):
    """A deployment can use Groq for chat while using fastembed for retrieval —
    the two provider knobs must not be coupled."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")

    assert isinstance(get_llm(), ChatGroq)
    assert isinstance(get_embeddings(), FastEmbedEmbeddings)


def test_get_embeddings_falls_back_to_llm_provider_when_unset(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    embeddings = get_embeddings()

    assert isinstance(embeddings, OllamaEmbeddings)
