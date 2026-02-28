"""Load USAspending award JSONL into a DuckDB database with a stable schema.

Usage:
    python scripts/build_duckdb.py [--input data/raw/usaspending/awards.jsonl]
                                   [--db data/processed/procurement.duckdb]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import duckdb

DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "usaspending" / "awards.jsonl"
)
DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "processed" / "procurement.duckdb"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS awards (
    internal_id          BIGINT,
    award_id             VARCHAR,
    recipient_name       VARCHAR,
    start_date           DATE,
    end_date             DATE,
    award_amount         DOUBLE,
    total_outlays        DOUBLE,
    description          VARCHAR,
    covid_obligations    DOUBLE,
    covid_outlays        DOUBLE,
    infra_obligations    DOUBLE,
    infra_outlays        DOUBLE,
    awarding_agency      VARCHAR,
    awarding_sub_agency  VARCHAR,
    contract_award_type  VARCHAR,
    recipient_id         VARCHAR,
    awarding_agency_id   INTEGER,
    agency_slug          VARCHAR,
    generated_internal_id VARCHAR
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_awards_agency ON awards (awarding_agency);",
    "CREATE INDEX IF NOT EXISTS idx_awards_recipient ON awards (recipient_name);",
    "CREATE INDEX IF NOT EXISTS idx_awards_start ON awards (start_date);",
    "CREATE INDEX IF NOT EXISTS idx_awards_type ON awards (contract_award_type);",
]


def _parse_record(raw: dict) -> dict:
    """Map raw API field names to clean column names."""
    return {
        "internal_id": raw.get("internal_id"),
        "award_id": raw.get("Award ID"),
        "recipient_name": raw.get("Recipient Name"),
        "start_date": raw.get("Start Date"),
        "end_date": raw.get("End Date"),
        "award_amount": raw.get("Award Amount"),
        "total_outlays": raw.get("Total Outlays"),
        "description": raw.get("Description"),
        "covid_obligations": raw.get("COVID-19 Obligations", 0),
        "covid_outlays": raw.get("COVID-19 Outlays", 0),
        "infra_obligations": raw.get("Infrastructure Obligations", 0),
        "infra_outlays": raw.get("Infrastructure Outlays", 0),
        "awarding_agency": raw.get("Awarding Agency"),
        "awarding_sub_agency": raw.get("Awarding Sub Agency"),
        "contract_award_type": raw.get("Contract Award Type"),
        "recipient_id": raw.get("recipient_id"),
        "awarding_agency_id": raw.get("awarding_agency_id"),
        "agency_slug": raw.get("agency_slug"),
        "generated_internal_id": raw.get("generated_internal_id"),
    }


def build_duckdb(input_path: Path = DEFAULT_INPUT, db_path: Path = DEFAULT_DB) -> Path:
    """Read JSONL and load into DuckDB."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing DB to rebuild from scratch
    if db_path.exists():
        db_path.unlink()

    records: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            records.append(_parse_record(json.loads(line)))

    logger.info("Parsed %d records from %s.", len(records), input_path)

    con = duckdb.connect(str(db_path))
    con.execute("DROP TABLE IF EXISTS awards;")
    con.execute(CREATE_TABLE_SQL)

    # Batch insert
    columns = list(records[0].keys())
    placeholders = ", ".join(["?"] * len(columns))
    insert_sql = f"INSERT INTO awards ({', '.join(columns)}) VALUES ({placeholders})"

    rows = [tuple(r[c] for c in columns) for r in records]
    con.executemany(insert_sql, rows)

    for idx_sql in CREATE_INDEXES_SQL:
        con.execute(idx_sql)

    count = con.execute("SELECT COUNT(*) FROM awards").fetchone()[0]  # type: ignore[index]
    logger.info("Loaded %d rows into %s.", count, db_path)

    con.close()
    return db_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DuckDB from awards JSONL")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    build_duckdb(input_path=args.input, db_path=args.db)


if __name__ == "__main__":
    main()
    sys.exit(0)
