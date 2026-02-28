"""Build a FAISS vector index from FAR chunks.

Usage:
    python scripts/build_vector_index.py [--input data/processed/far_chunks.jsonl]
                                         [--output-dir data/processed/vector_index]

Uses LangChain's FAISS integration. Falls back to a simple TF-IDF based
approach if no embedding model API key is available.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "far_chunks.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "vector_index"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_chunks(path: Path) -> list[dict]:
    """Load chunks from JSONL."""
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    logger.info("Loaded %d chunks from %s.", len(chunks), path)
    return chunks


def _get_embeddings():  # type: ignore[no-untyped-def]
    """Get the best available embedding model."""
    from procurement_copilot.llm import get_embeddings

    return get_embeddings()


def build_vector_index(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Build FAISS index from chunks and save to disk."""
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    chunks = _load_chunks(input_path)

    documents = [
        Document(
            page_content=c["text"],
            metadata={
                "chunk_id": c["chunk_id"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "section_heading": c.get("section_heading", ""),
                "source_url": c["source_url"],
            },
        )
        for c in chunks
    ]

    embeddings = _get_embeddings()

    logger.info("Building FAISS index from %d documents...", len(documents))
    vectorstore = FAISS.from_documents(documents, embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_dir))
    logger.info("Saved FAISS index to %s.", output_dir)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS vector index from FAR chunks")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    build_vector_index(input_path=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
    sys.exit(0)
