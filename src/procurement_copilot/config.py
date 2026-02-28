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

    # --- Paths ---
    DATA_DIR: Path = _PROJECT_ROOT / "data"
    DB_PATH: Path = _PROJECT_ROOT / "data" / "processed" / "procurement.duckdb"
    VECTOR_INDEX_DIR: Path = _PROJECT_ROOT / "data" / "processed" / "vector_index"
    KG_PATH: Path = _PROJECT_ROOT / "data" / "processed" / "kg.parquet"

    # --- Runtime ---
    LOG_LEVEL: str = "INFO"


settings = Settings()
