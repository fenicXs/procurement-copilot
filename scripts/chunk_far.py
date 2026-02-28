"""Extract text from FAR PDF and chunk into overlapping segments with metadata.

Usage:
    python scripts/chunk_far.py [--input data/raw/far/FAR.pdf]
                                [--output data/processed/far_chunks.jsonl]

Each chunk includes: chunk_id, text, page_start, page_end, source_url, and
an inferred section heading when available.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from pypdf import PdfReader

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "far" / "FAR.pdf"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "far_chunks.jsonl"

CHUNK_SIZE = 1000  # target characters per chunk
CHUNK_OVERLAP = 200
SOURCE_URL = "https://www.acquisition.gov/far"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Pattern to detect FAR section headings like "Part 1", "Subpart 1.1", "1.101", "52.204-1"
_HEADING_RE = re.compile(
    r"^((?:Part|Subpart)\s+\d+[\.\d]*" r"|(?:\d{1,2}\.\d{2,4}(?:-\d+)?)" r"|(?:Section\s+\d+))",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract (page_number, text) pairs from a PDF."""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Clean up common PDF artifacts
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i + 1, text))  # 1-indexed page numbers
    logger.info("Extracted text from %d pages.", len(pages))
    return pages


def _detect_heading(text: str) -> str:
    """Try to detect a FAR section heading from the start of a text block."""
    match = _HEADING_RE.search(text[:200])
    return match.group(1).strip() if match else ""


def chunk_far(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Path:
    """Chunk the FAR PDF into overlapping text segments with metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pages = _extract_pages(input_path)

    # Build a flat list of (page_number, text_segment) for chunking
    # We concatenate all page texts with page markers
    page_texts: list[dict] = []
    for page_num, text in pages:
        page_texts.append({"page": page_num, "text": text})

    # Sliding window chunking across the full document
    chunks: list[dict] = []
    chunk_id = 0

    # Concatenate all text with page boundary markers
    full_text = ""
    page_boundaries: list[tuple[int, int, int]] = []  # (start_char, end_char, page_num)

    for pt in page_texts:
        start = len(full_text)
        full_text += pt["text"] + " "
        end = len(full_text)
        page_boundaries.append((start, end, pt["page"]))

    # Sliding window
    pos = 0
    while pos < len(full_text):
        end = min(pos + chunk_size, len(full_text))
        chunk_text = full_text[pos:end].strip()

        if not chunk_text:
            break

        # Determine which pages this chunk spans
        page_start = None
        page_end = None
        for bstart, bend, pnum in page_boundaries:
            if bstart < end and bend > pos:
                if page_start is None:
                    page_start = pnum
                page_end = pnum

        heading = _detect_heading(chunk_text)

        chunks.append(
            {
                "chunk_id": f"far_chunk_{chunk_id:05d}",
                "text": chunk_text,
                "page_start": page_start or 1,
                "page_end": page_end or page_start or 1,
                "section_heading": heading,
                "source_url": SOURCE_URL,
            }
        )
        chunk_id += 1

        # Advance by (chunk_size - overlap)
        pos += chunk_size - chunk_overlap
        if end >= len(full_text):
            break

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    logger.info(
        "Created %d chunks (avg %d chars) -> %s",
        len(chunks),
        sum(len(c["text"]) for c in chunks) // max(len(chunks), 1),
        output_path,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk FAR PDF into JSONL")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()
    chunk_far(args.input, args.output, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
    sys.exit(0)
