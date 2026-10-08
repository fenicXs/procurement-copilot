"""Application settings loaded from environment variables or .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider keys (at least one required for real inference) ---
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    BYTEZ_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # --- Explicit provider override: "openai"|"anthropic"|"bytez"|"ollama"|"groq". ---
    # Empty = auto fallback chain.
    LLM_PROVIDER: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"

    # --- Embedding provider override, independent of LLM_PROVIDER: lets a
    # deployment use a hosted LLM (e.g. "groq") for generation while using
    # free CPU-only "fastembed" embeddings — no GPU, no API key needed for
    # retrieval. Falls back to LLM_PROVIDER's chain when unset. ---
    EMBEDDING_PROVIDER: str = ""
    FASTEMBED_DENSE_MODEL: str = "BAAI/bge-small-en-v1.5"

    # --- Ollama (local, free) ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma4:31b-it-bf16"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    # Ollama defaults num_ctx to the model's full max context (262144 for
    # gemma4:31b) when unset, which needs ~111GB KV-cache+compute-graph memory
    # and runs right at the edge of an 80GB GPU. Our RAG prompts are a few
    # ~1000-char chunks + a query — cap it well below that ceiling.
    OLLAMA_NUM_CTX: int = 8192

    # --- LangFuse (observability — Cloud free tier) ---
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # --- Vector retrieval (Qdrant local-mode hybrid dense+sparse) ---
    QDRANT_COLLECTION: str = "far_chunks"
    SPARSE_MODEL: str = "Qdrant/bm25"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    # ONNX reranker used when EMBEDDING_PROVIDER=fastembed — avoids pulling
    # torch/sentence-transformers (a huge dependency, the actual cause of an
    # OOM crash on a 512MB free-tier deploy) into the lightweight CPU path.
    FASTEMBED_RERANKER_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Optional second rerank stage (fastembed path only): the small model above
    # shortlists, then this stronger one re-scores just the shortlist. Fixes
    # defining chunks (e.g. FAR 6.302-1) that MiniLM ranks below chunks that
    # merely cite them. Empty string disables stage 2. OFF by default: on the
    # 4GiB Cloud Run instance bge-reranker-v2-m3-int8 over a 25-chunk shortlist
    # was OOM-killed (4.2-4.3GiB used) — see git history before re-enabling.
    FASTEMBED_RERANKER2_MODEL: str = ""
    RERANK_SHORTLIST: int = 25

    # --- Evaluation (RAGAS) ---
    # Forces the local, free judge regardless of which provider answers
    # queries, so eval cost stays zero even if a cloud key is configured.
    EVAL_JUDGE_PROVIDER: str = "ollama"

    # --- Paths ---
    DATA_DIR: Path = _PROJECT_ROOT / "data"
    DB_PATH: Path = _PROJECT_ROOT / "data" / "processed" / "procurement.duckdb"
    VECTOR_INDEX_DIR: Path = _PROJECT_ROOT / "data" / "processed" / "vector_index"
    KG_PATH: Path = _PROJECT_ROOT / "data" / "processed" / "kg.parquet"
    AUDIT_LOG_PATH: Path = _PROJECT_ROOT / "logs" / "audit.jsonl"

    # --- Runtime ---
    LOG_LEVEL: str = "INFO"


settings = Settings()
