"""SQL safety layer: validate, sanitize, and execute queries against DuckDB.

Safety constraints:
- Read-only: only SELECT statements allowed
- Allowlisted tables only
- Dangerous keywords blocked (DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, etc.)
- LIMIT enforced on every query (default 1000)
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from procurement_copilot.config import settings

logger = logging.getLogger(__name__)

ALLOWED_TABLES: frozenset[str] = frozenset({"awards"})
DEFAULT_LIMIT = 1000

BLOCKED_KEYWORDS: frozenset[str] = frozenset(
    {
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "REPLACE",
        "GRANT",
        "REVOKE",
        "EXEC",
        "EXECUTE",
        "ATTACH",
        "DETACH",
        "COPY",
        "EXPORT",
        "IMPORT",
        "LOAD",
        "INSTALL",
    }
)

# Regex to detect blocked keywords as standalone tokens (case-insensitive)
_BLOCKED_RE = re.compile(
    r"\b(" + "|".join(BLOCKED_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Regex to detect existing LIMIT clause
_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


class SQLValidationError(Exception):
    """Raised when a SQL query fails safety validation."""


@dataclass
class SQLResult:
    """Result of a safe SQL execution."""

    sql: str
    columns: list[str]
    rows: list[tuple]
    row_count: int
    error: str | None = None
    metadata: dict = field(default_factory=dict)


def validate_sql(sql: str, allowed_tables: frozenset[str] = ALLOWED_TABLES) -> str:
    """Validate and sanitize a SQL query. Returns the (possibly modified) query.

    Raises SQLValidationError if the query is unsafe.
    """
    stripped = sql.strip().rstrip(";")

    # 1. Must start with SELECT
    if not stripped.upper().startswith("SELECT"):
        raise SQLValidationError("Only SELECT queries are allowed.")

    # 2. Check for blocked keywords
    match = _BLOCKED_RE.search(stripped)
    if match:
        raise SQLValidationError(f"Blocked keyword detected: {match.group(0).upper()}")

    # 3. Check that only allowed tables are referenced
    # Extract table names after FROM and JOIN keywords
    table_pattern = re.compile(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
    referenced_tables = {t.lower() for t in table_pattern.findall(stripped)}
    disallowed = referenced_tables - allowed_tables
    if disallowed:
        raise SQLValidationError(
            f"Query references disallowed table(s): {', '.join(sorted(disallowed))}"
        )

    # 4. Enforce LIMIT
    if not _LIMIT_RE.search(stripped):
        stripped = f"{stripped} LIMIT {DEFAULT_LIMIT}"

    return stripped


def execute_safe_sql(
    sql: str,
    db_path: Path | None = None,
    allowed_tables: frozenset[str] = ALLOWED_TABLES,
) -> SQLResult:
    """Validate and execute a SQL query, returning structured results."""
    db = db_path or settings.DB_PATH

    try:
        validated_sql = validate_sql(sql, allowed_tables)
    except SQLValidationError as exc:
        return SQLResult(sql=sql, columns=[], rows=[], row_count=0, error=str(exc))

    try:
        con = duckdb.connect(str(db), read_only=True)
        result = con.execute(validated_sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        con.close()

        return SQLResult(
            sql=validated_sql,
            columns=columns,
            rows=rows,
            row_count=len(rows),
        )
    except Exception as exc:
        logger.exception("SQL execution error")
        return SQLResult(
            sql=validated_sql,
            columns=[],
            rows=[],
            row_count=0,
            error=f"Execution error: {exc}",
        )


def get_schema(db_path: Path | None = None) -> dict[str, list[dict]]:
    """Return the schema of all allowed tables as {table: [{name, type}, ...]}."""
    db = db_path or settings.DB_PATH
    con = duckdb.connect(str(db), read_only=True)
    schema: dict[str, list[dict]] = {}

    for table in ALLOWED_TABLES:
        try:
            cols = con.execute(
                f"SELECT column_name, data_type FROM information_schema.columns "
                f"WHERE table_name = '{table}' ORDER BY ordinal_position"
            ).fetchall()
            schema[table] = [{"name": c[0], "type": c[1]} for c in cols]
        except Exception:
            schema[table] = []

    con.close()
    return schema
