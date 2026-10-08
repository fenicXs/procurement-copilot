"""LLM prompts live as plain .txt files next to this module so they can be
tuned without touching Python. Edit a file, push, and CI redeploys.

Prompts that need runtime values use `str.format` placeholders (e.g. `{schema}`
in sql_generate.txt) — callers fill them with `.format(...)`, so any literal
brace in those files must be doubled.
"""

from functools import lru_cache
from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Return the text of prompts/<name>.txt (surrounding whitespace stripped)."""
    path = _PROMPT_DIR / f"{name}.txt"
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {path}") from None
