"""Fetch award-level spend data from the USAspending API.

Usage:
    python scripts/fetch_usaspending_awards.py [--rows 5000] [--output-dir data/raw/usaspending]

Paginates through the spending_by_award endpoint and writes JSONL output.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import httpx

API_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "usaspending"
DEFAULT_ROWS = 5000
PAGE_SIZE = 100  # API max per request
MAX_RETRIES = 3
TIMEOUT_SECONDS = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Filter: recent contract awards with non-trivial value
REQUEST_PAYLOAD_TEMPLATE: dict = {
    "filters": {
        "time_period": [{"start_date": "2023-01-01", "end_date": "2024-12-31"}],
        "award_type_codes": ["A", "B", "C", "D"],  # Contract types
    },
    "fields": [
        "Award ID",
        "Recipient Name",
        "Start Date",
        "End Date",
        "Award Amount",
        "Total Outlays",
        "Description",
        "def_codes",
        "COVID-19 Obligations",
        "COVID-19 Outlays",
        "Infrastructure Obligations",
        "Infrastructure Outlays",
        "Awarding Agency",
        "Awarding Sub Agency",
        "Contract Award Type",
        "recipient_id",
        "prime_award_recipient_id",
    ],
    "limit": PAGE_SIZE,
    "page": 1,
    "subawards": False,
    "order": "desc",
    "sort": "Award Amount",
}


def fetch_usaspending_awards(
    target_rows: int = DEFAULT_ROWS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Paginate through the USAspending API and save results as JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "awards.jsonl"

    if dest.exists():
        existing = sum(1 for _ in open(dest))
        if existing >= target_rows:
            logger.info(
                "awards.jsonl already has %d rows (target %d) — skipping.",
                existing,
                target_rows,
            )
            return dest

    collected: list[dict] = []
    page = 1

    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        while len(collected) < target_rows:
            payload = {**REQUEST_PAYLOAD_TEMPLATE, "page": page}

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    logger.info(
                        "Fetching page %d (collected %d / %d)...",
                        page,
                        len(collected),
                        target_rows,
                    )
                    resp = client.post(API_URL, json=payload)
                    resp.raise_for_status()
                    break
                except (httpx.HTTPError, httpx.TransportError) as exc:
                    logger.warning("Page %d attempt %d failed: %s", page, attempt, exc)
                    if attempt < MAX_RETRIES:
                        time.sleep(2**attempt)
                    else:
                        raise

            data = resp.json()
            results = data.get("results", [])

            if not results:
                logger.info("No more results from API at page %d.", page)
                break

            collected.extend(results)
            page += 1

            # Be a good API citizen — small delay between pages
            time.sleep(0.3)

    # Trim to exact target
    collected = collected[:target_rows]

    with open(dest, "w", encoding="utf-8") as f:
        for record in collected:
            f.write(json.dumps(record, default=str) + "\n")

    logger.info("Wrote %d award records to %s.", len(collected), dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch USAspending award data")
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help=f"Number of rows to fetch (default: {DEFAULT_ROWS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save awards.jsonl",
    )
    args = parser.parse_args()
    fetch_usaspending_awards(target_rows=args.rows, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
    sys.exit(0)
