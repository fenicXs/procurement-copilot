import json
from pathlib import Path

from procurement_copilot.audit_log import log_query


class TestLogQuery:
    def test_writes_one_jsonl_record(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        log_query(
            question="What is a simplified acquisition threshold?",
            intent="rag",
            is_verified=True,
            abstained=False,
            citation_chunk_ids=["far_13.1_001"],
            session_id="sess-1",
            log_path=log_path,
        )

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["question"] == "What is a simplified acquisition threshold?"
        assert record["intent"] == "rag"
        assert record["is_verified"] is True
        assert record["abstained"] is False
        assert record["citation_chunk_ids"] == ["far_13.1_001"]
        assert record["session_id"] == "sess-1"
        assert "timestamp" in record

    def test_appends_across_calls(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        for i in range(3):
            log_query(
                question=f"question {i}",
                intent="rag",
                is_verified=True,
                abstained=False,
                citation_chunk_ids=[],
                log_path=log_path,
            )

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        log_path = tmp_path / "nested" / "dir" / "audit.jsonl"
        log_query(
            question="q",
            intent="rag",
            is_verified=False,
            abstained=True,
            citation_chunk_ids=[],
            log_path=log_path,
        )
        assert log_path.exists()

    def test_does_not_raise_on_unwritable_path(self, tmp_path: Path) -> None:
        # Parent is a file, not a directory — mkdir will fail, and log_query
        # must swallow the error rather than propagate it.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        log_path = blocker / "audit.jsonl"

        log_query(
            question="q",
            intent="rag",
            is_verified=True,
            abstained=False,
            citation_chunk_ids=[],
            log_path=log_path,
        )
