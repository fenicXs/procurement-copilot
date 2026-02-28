"""Tests for SQL safety layer and DuckDB schema."""

from pathlib import Path

import duckdb
import pytest

from procurement_copilot.sql.safe_sql import (
    SQLValidationError,
    execute_safe_sql,
    get_schema,
    validate_sql,
)

# --- validate_sql unit tests ---


class TestValidateSQL:
    def test_select_allowed(self) -> None:
        result = validate_sql("SELECT * FROM awards")
        assert result.upper().startswith("SELECT")

    def test_drop_blocked(self) -> None:
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validate_sql("DROP TABLE awards")

    def test_delete_blocked(self) -> None:
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validate_sql("DELETE FROM awards WHERE 1=1")

    def test_update_blocked(self) -> None:
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validate_sql("UPDATE awards SET award_amount = 0")

    def test_insert_blocked(self) -> None:
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validate_sql("INSERT INTO awards VALUES (1)")

    def test_blocked_keyword_in_subquery(self) -> None:
        """Blocked keywords caught even when embedded in a SELECT."""
        with pytest.raises(SQLValidationError, match="DROP"):
            validate_sql("SELECT * FROM awards; DROP TABLE awards")

    def test_non_select_blocked(self) -> None:
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validate_sql("SHOW TABLES")

    def test_disallowed_table(self) -> None:
        with pytest.raises(SQLValidationError, match="disallowed table"):
            validate_sql("SELECT * FROM secret_table")

    def test_limit_injected(self) -> None:
        result = validate_sql("SELECT * FROM awards")
        assert "LIMIT" in result.upper()

    def test_existing_limit_preserved(self) -> None:
        result = validate_sql("SELECT * FROM awards LIMIT 10")
        # Should not double-add LIMIT
        assert result.upper().count("LIMIT") == 1

    def test_semicolon_stripped(self) -> None:
        result = validate_sql("SELECT * FROM awards;")
        assert not result.endswith(";")

    def test_join_table_checked(self) -> None:
        with pytest.raises(SQLValidationError, match="disallowed table"):
            validate_sql("SELECT * FROM awards JOIN users ON 1=1")

    def test_case_insensitive_block(self) -> None:
        with pytest.raises(SQLValidationError, match="DROP"):
            validate_sql("select * from awards; drop table awards")


# --- execute_safe_sql integration tests (with a temp DuckDB) ---


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Create a temporary DuckDB with sample data."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE awards (
            internal_id BIGINT,
            award_id VARCHAR,
            recipient_name VARCHAR,
            start_date DATE,
            end_date DATE,
            award_amount DOUBLE,
            total_outlays DOUBLE,
            description VARCHAR,
            covid_obligations DOUBLE,
            covid_outlays DOUBLE,
            infra_obligations DOUBLE,
            infra_outlays DOUBLE,
            awarding_agency VARCHAR,
            awarding_sub_agency VARCHAR,
            contract_award_type VARCHAR,
            recipient_id VARCHAR,
            awarding_agency_id INTEGER,
            agency_slug VARCHAR,
            generated_internal_id VARCHAR
        );
    """
    )
    con.execute(
        """
        INSERT INTO awards VALUES
        (1, 'AWD-001', 'ACME CORP', '2023-01-15', '2024-01-15',
         1000000.0, 500000.0, 'Test contract', 0, 0, 0, 0,
         'Department of Defense', 'Army', 'DEFINITIVE CONTRACT',
         'r1', 100, 'dod', 'gen-1'),
        (2, 'AWD-002', 'GLOBEX INC', '2023-06-01', '2024-06-01',
         2500000.0, 1200000.0, 'IT services', 0, 0, 0, 0,
         'Department of Health and Human Services', 'CDC', 'BPA CALL',
         'r2', 200, 'hhs', 'gen-2'),
        (3, 'AWD-003', 'ACME CORP', '2024-01-01', '2024-12-31',
         750000.0, 300000.0, 'Maintenance', 0, 0, 0, 0,
         'Department of Defense', 'Navy', 'PURCHASE ORDER',
         'r3', 100, 'dod', 'gen-3');
    """
    )
    con.close()
    return db_path


class TestExecuteSafeSQL:
    def test_select_returns_results(self, tmp_db: Path) -> None:
        result = execute_safe_sql("SELECT * FROM awards", db_path=tmp_db)
        assert result.error is None
        assert result.row_count == 3
        assert "award_id" in result.columns

    def test_aggregate_query(self, tmp_db: Path) -> None:
        result = execute_safe_sql(
            "SELECT awarding_agency, SUM(award_amount) as total "
            "FROM awards GROUP BY awarding_agency",
            db_path=tmp_db,
        )
        assert result.error is None
        assert result.row_count == 2  # DoD and HHS

    def test_blocked_query_returns_error(self, tmp_db: Path) -> None:
        result = execute_safe_sql("DROP TABLE awards", db_path=tmp_db)
        assert result.error is not None
        assert "Only SELECT" in result.error

    def test_disallowed_table_returns_error(self, tmp_db: Path) -> None:
        result = execute_safe_sql("SELECT * FROM secrets", db_path=tmp_db)
        assert result.error is not None
        assert "disallowed" in result.error


class TestGetSchema:
    def test_schema_returns_columns(self, tmp_db: Path) -> None:
        schema = get_schema(db_path=tmp_db)
        assert "awards" in schema
        col_names = [c["name"] for c in schema["awards"]]
        assert "award_id" in col_names
        assert "award_amount" in col_names
        assert "awarding_agency" in col_names
