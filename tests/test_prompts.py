"""Prompts live in src/procurement_copilot/prompts/*.txt — guard the loader."""

import pytest

from procurement_copilot.prompts import _PROMPT_DIR, load_prompt

EXPECTED_PROMPTS = {
    "verify",
    "intent",
    "grade",
    "rewrite",
    "rag_answer",
    "graph_answer",
    "sql_generate",
    "sql_summarize",
    "merge_answers",
}


def test_every_expected_prompt_file_exists_and_is_non_empty() -> None:
    on_disk = {p.stem for p in _PROMPT_DIR.glob("*.txt")}
    assert EXPECTED_PROMPTS <= on_disk
    for name in EXPECTED_PROMPTS:
        assert load_prompt(name).strip(), f"{name}.txt is empty"


def test_sql_prompt_formats_with_only_the_schema_placeholder() -> None:
    # A stray literal brace in the .txt would raise KeyError/ValueError here.
    rendered = load_prompt("sql_generate").format(schema="awards: award_id (VARCHAR)")
    assert "awards: award_id (VARCHAR)" in rendered
    assert "{" not in rendered


def test_verifier_prompt_keeps_the_machine_readable_format() -> None:
    # verifier.py parses these exact markers out of the judge's reply.
    text = load_prompt("verify")
    assert "GROUNDED:" in text and "UNSUPPORTED:" in text


def test_module_constants_are_loaded_from_the_files() -> None:
    from procurement_copilot.orchestrator.intent import INTENT_SYSTEM_PROMPT
    from procurement_copilot.rag.crag import GRADE_SYSTEM_PROMPT

    assert INTENT_SYSTEM_PROMPT == load_prompt("intent")
    assert GRADE_SYSTEM_PROMPT == load_prompt("grade")


def test_missing_prompt_raises_a_clear_error() -> None:
    with pytest.raises(FileNotFoundError, match="nonexistent_prompt"):
        load_prompt("nonexistent_prompt")
