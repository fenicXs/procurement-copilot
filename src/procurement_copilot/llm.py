"""LLM provider utilities — shared across RAG, Graph, and SQL answer modules.

Fallback chain: OpenAI → Anthropic → Bytez → Ollama → None.
Set LLM_PROVIDER to force a specific provider regardless of which keys are present.
"""

import logging
import os
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from procurement_copilot.config import settings

logger = logging.getLogger(__name__)


def _get_key(name: str) -> str:
    """Get an API key from os.environ first, then fall back to pydantic settings."""
    return os.environ.get(name, "") or getattr(settings, name, "")


class ChatBytez(BaseChatModel):
    """Minimal LangChain-compatible wrapper around the Bytez Python SDK."""

    api_key: str = ""
    model_name: str = "openai/gpt-4.1-mini"
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "bytez"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Any:
        from bytez import Bytez
        from langchain_core.outputs import ChatGeneration, ChatResult

        sdk = Bytez(self.api_key)
        model = sdk.model(self.model_name)

        # Convert LangChain messages to OpenAI-style dicts
        msg_dicts = []
        for m in messages:
            if m.type == "system":
                msg_dicts.append({"role": "system", "content": m.content})
            elif m.type == "human":
                msg_dicts.append({"role": "user", "content": m.content})
            elif m.type == "ai":
                msg_dicts.append({"role": "assistant", "content": m.content})
            else:
                msg_dicts.append({"role": "user", "content": m.content})

        result = model.run(msg_dicts)

        if result.error:
            raise RuntimeError(f"Bytez API error: {result.error}")

        if isinstance(result.output, dict):
            content = result.output.get("content", "")
        else:
            content = str(result.output)

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def _ollama_chat() -> BaseChatModel:
    from langchain_ollama import ChatOllama

    logger.info("Using Ollama LLM provider (%s).", settings.OLLAMA_MODEL)
    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=0,
        num_ctx=settings.OLLAMA_NUM_CTX,
    )


def _groq_chat() -> BaseChatModel:
    from langchain_groq import ChatGroq
    from pydantic import SecretStr

    logger.info("Using Groq LLM provider (%s).", settings.GROQ_MODEL)
    return ChatGroq(
        api_key=SecretStr(_get_key("GROQ_API_KEY")),
        model=settings.GROQ_MODEL,
        temperature=0,
    )


def get_llm() -> BaseChatModel | None:
    """Get the best available chat model.

    Fallback chain: OpenAI → Anthropic → Bytez → None.
    Set LLM_PROVIDER ("openai"|"anthropic"|"bytez"|"ollama"|"groq") to force a
    specific provider — this is the only way to opt into Ollama or Groq, since
    neither is auto-detected, so offline/CI test runs never pick them up
    silently.
    """
    provider = (_get_key("LLM_PROVIDER") or settings.LLM_PROVIDER).lower()

    if provider == "ollama":
        return _ollama_chat()

    if provider == "groq":
        return _groq_chat()

    if provider in ("", "openai") and _get_key("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        logger.info("Using OpenAI LLM provider.")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if provider in ("", "anthropic") and _get_key("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        logger.info("Using Anthropic LLM provider.")
        # `model` is accepted at runtime via a pydantic alias for `model_name`;
        # installed langchain-anthropic's stubs don't reflect that alias.
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)  # type: ignore[call-arg]

    bytez_key = _get_key("BYTEZ_API_KEY")
    if provider in ("", "bytez") and bytez_key:
        logger.info("Using Bytez LLM provider.")
        return ChatBytez(
            api_key=bytez_key,
            model_name="openai/gpt-4.1-mini",
            temperature=0,
        )

    return None


class BytezEmbeddings(Embeddings):
    """LangChain-compatible embeddings using Bytez SDK."""

    def __init__(self, api_key: str, model_name: str = "openai/text-embedding-3-small"):
        self.api_key = api_key
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from bytez import Bytez

        def _embed_one(idx_text: tuple[int, str]) -> tuple[int, list[float]]:
            idx, text = idx_text
            for attempt in range(5):
                sdk = Bytez(self.api_key)
                model = sdk.model(self.model_name)
                result = model.run(text)
                # Handle both Response objects and raw list returns
                if isinstance(result, list):
                    return (idx, result)
                if hasattr(result, "error") and result.error:
                    if "rate limit" in str(result.error).lower():
                        time.sleep(2 * (attempt + 1))
                        continue
                    raise RuntimeError(f"Bytez embedding error: {result.error}")
                return (idx, result.output)
            raise RuntimeError(f"Bytez rate limit exceeded after retries for text {idx}")

        results: list[tuple[int, list[float]]] = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(_embed_one, (i, t)): i for i, t in enumerate(texts)}
            done = 0
            for future in as_completed(futures):
                results.append(future.result())
                done += 1
                if done % 100 == 0:
                    logger.info("Embedded %d / %d texts.", done, len(texts))

        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text — direct call, no thread pool."""
        import time

        from bytez import Bytez

        for attempt in range(5):
            sdk = Bytez(self.api_key)
            model = sdk.model(self.model_name)
            result = model.run(text)
            if isinstance(result, list):
                return result
            if hasattr(result, "error") and result.error:
                if "rate limit" in str(result.error).lower():
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(f"Bytez embedding error: {result.error}")
            return result.output  # type: ignore[no-any-return]  # Bytez SDK result is untyped
        raise RuntimeError("Bytez rate limit exceeded after retries")


def _ollama_embeddings() -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    logger.info("Using Ollama embeddings (%s).", settings.OLLAMA_EMBED_MODEL)
    return OllamaEmbeddings(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_EMBED_MODEL)


class FastEmbedEmbeddings(Embeddings):
    """CPU-only dense embeddings via `fastembed` (ONNX runtime) — no GPU, no
    API key. Makes the Qdrant index and query path fully portable to
    environments with neither Ollama nor a hosted embeddings key (e.g. a
    HuggingFace Space). Lazily loads the model once per process.

    `threads` is passed explicitly rather than left to ONNX Runtime's
    default: on a cgroup-limited allocation (e.g. a 4-CPU SLURM job on a
    many-core node), onnxruntime's default thread pool is sized off the
    full host, not the cgroup quota — it then tries to pin far more threads
    than there are real cores, which both floods the log with
    `pthread_setaffinity_np failed` errors and causes severe contention
    (observed: ~2 min/batch of 64 chunks instead of a few seconds).
    """

    def __init__(self, model_name: str, threads: int | None = None):
        self.model_name = model_name
        # os.cpu_count() reports the full host's core count, ignoring cgroup
        # CPU quotas (e.g. a 4-CPU SLURM allocation on a 128-core node) —
        # sched_getaffinity reflects the process's actual usable cores.
        self.threads = threads or len(os.sched_getaffinity(0))
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name, threads=self.threads)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._get_model().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._get_model().embed([text]))).tolist()  # type: ignore[no-any-return]


_fastembed_singleton: FastEmbedEmbeddings | None = None


def _fastembed_embeddings() -> Embeddings:
    global _fastembed_singleton
    if _fastembed_singleton is None:
        logger.info("Using fastembed dense embeddings (%s).", settings.FASTEMBED_DENSE_MODEL)
        _fastembed_singleton = FastEmbedEmbeddings(settings.FASTEMBED_DENSE_MODEL)
    return _fastembed_singleton


def get_embeddings() -> Embeddings:
    """Get the best available embedding model.

    `EMBEDDING_PROVIDER` is checked first and is independent of
    `LLM_PROVIDER` — this lets a deployment use a hosted chat model (e.g.
    Groq) for generation while still using free, local, CPU-only fastembed
    embeddings for retrieval. Falls back to `LLM_PROVIDER`'s chain when unset.
    Final fallback chain: OpenAI → Bytez → FakeEmbeddings.
    """
    provider = (
        _get_key("EMBEDDING_PROVIDER")
        or settings.EMBEDDING_PROVIDER
        or _get_key("LLM_PROVIDER")
        or settings.LLM_PROVIDER
    ).lower()

    if provider == "fastembed":
        return _fastembed_embeddings()

    if provider == "ollama":
        return _ollama_embeddings()

    if provider in ("", "openai") and _get_key("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings

        logger.info("Using OpenAI embeddings.")
        return OpenAIEmbeddings(model="text-embedding-3-small")

    bytez_key = _get_key("BYTEZ_API_KEY")
    if provider in ("", "bytez") and bytez_key:
        logger.info("Using Bytez embeddings.")
        return BytezEmbeddings(api_key=bytez_key)

    from langchain_community.embeddings import FakeEmbeddings

    logger.warning("No embedding provider — using FakeEmbeddings.")
    return FakeEmbeddings(size=1536)
