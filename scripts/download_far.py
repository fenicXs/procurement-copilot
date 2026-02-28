"""Download the FAR (Federal Acquisition Regulation) PDF from Acquisition.gov.

Usage:
    python scripts/download_far.py [--output-dir data/raw/far]

The script downloads with retries and timeout, and skips if the file already exists.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import httpx

FAR_URL = "https://www.acquisition.gov/sites/default/files/current/far/pdf/FAR.pdf"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "far"
MAX_RETRIES = 3
TIMEOUT_SECONDS = 300  # 5 minutes — the PDF is large (~14 MB)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def download_far(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Download FAR.pdf to *output_dir* and return the file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "FAR.pdf"

    if dest.exists() and dest.stat().st_size > 1_000_000:
        size = dest.stat().st_size
        logger.info("FAR.pdf already exists at %s (%d bytes) — skipping.", dest, size)
        return dest

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info("Downloading FAR PDF (attempt %d/%d) from %s", attempt, MAX_RETRIES, FAR_URL)
        try:
            with httpx.stream(
                "GET", FAR_URL, timeout=TIMEOUT_SECONDS, follow_redirects=True
            ) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with open(dest, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65_536):
                        f.write(chunk)
                        downloaded += len(chunk)
                if total and downloaded < total:
                    raise httpx.TransportError(f"Incomplete download: {downloaded}/{total} bytes")
            logger.info("Saved FAR.pdf to %s (%d bytes).", dest, dest.stat().st_size)
            return dest
        except (httpx.HTTPError, httpx.TransportError) as exc:
            logger.warning("Attempt %d failed: %s", attempt, exc)
            if attempt < MAX_RETRIES:
                wait = 2**attempt
                logger.info("Retrying in %d seconds...", wait)
                time.sleep(wait)
            else:
                logger.error("All %d attempts failed.", MAX_RETRIES)
                raise

    raise RuntimeError("Download failed after all retries")  # unreachable, satisfies mypy


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FAR PDF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save FAR.pdf",
    )
    args = parser.parse_args()
    download_far(args.output_dir)


if __name__ == "__main__":
    main()
    sys.exit(0)
