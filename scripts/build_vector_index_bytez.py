"""Build FAISS index using Bytez embeddings with batching and checkpointing.

Embeds chunks sequentially to respect Bytez API rate limits, saving progress
every 500 chunks so we can resume if interrupted.

Usage:
    BYTEZ_API_KEY=... python scripts/build_vector_index_bytez.py
"""

import json
import logging
import pickle
import sys
import time
from pathlib import Path

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "far_chunks.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "vector_index"
CHECKPOINT_PATH = OUTPUT_DIR / "_embeddings_checkpoint.pkl"
BYTEZ_MODEL = "openai/text-embedding-3-small"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_chunks() -> list[dict]:
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    logger.info("Loaded %d chunks.", len(chunks))
    return chunks


def embed_with_checkpoint(chunks: list[dict], api_key: str) -> list[list[float]]:
    """Embed all chunks, resuming from checkpoint if available."""
    from bytez import Bytez

    # Load checkpoint if exists
    start_idx = 0
    embeddings: list[list[float]] = []
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "rb") as f:
            embeddings = pickle.load(f)
        start_idx = len(embeddings)
        logger.info("Resumed from checkpoint: %d / %d already embedded.", start_idx, len(chunks))

    if start_idx >= len(chunks):
        logger.info("All chunks already embedded.")
        return embeddings

    sdk = Bytez(api_key)
    model = sdk.model(BYTEZ_MODEL)

    for i in range(start_idx, len(chunks)):
        text = chunks[i]["text"]

        # Retry with backoff
        for attempt in range(5):
            result = model.run(text)
            # Handle both Response objects and raw list returns
            if isinstance(result, list):
                embeddings.append(result)
                break
            if hasattr(result, "error") and result.error:
                err = str(result.error).lower()
                if "rate limit" in err:
                    wait = 3 * (attempt + 1)
                    logger.warning("Rate limited at chunk %d, waiting %ds...", i, wait)
                    time.sleep(wait)
                    # Re-create SDK connection after rate limit
                    sdk = Bytez(api_key)
                    model = sdk.model(BYTEZ_MODEL)
                    continue
                raise RuntimeError(f"Bytez error at chunk {i}: {result.error}")
            embeddings.append(result.output)
            break
        else:
            raise RuntimeError(f"Failed after 5 retries at chunk {i}")

        # Progress + checkpoint
        if (i + 1) % 100 == 0:
            logger.info("Embedded %d / %d chunks.", i + 1, len(chunks))
        if (i + 1) % 500 == 0:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(CHECKPOINT_PATH, "wb") as f:
                pickle.dump(embeddings, f)
            logger.info("Checkpoint saved at %d chunks.", i + 1)

    # Final checkpoint
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info("All %d chunks embedded.", len(embeddings))
    return embeddings


def build_index(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Build FAISS index from pre-computed embeddings and save."""
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    from procurement_copilot.llm import BytezEmbeddings

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

    # Build FAISS from pre-computed embeddings
    text_embeddings = list(zip([d.page_content for d in documents], embeddings))

    # Use a dummy BytezEmbeddings for the vectorstore (needed for query-time embedding)
    import os

    embed_model = BytezEmbeddings(api_key=os.environ["BYTEZ_API_KEY"])

    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embed_model,
        metadatas=[d.metadata for d in documents],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(OUTPUT_DIR))
    logger.info("FAISS index saved to %s.", OUTPUT_DIR)

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info("Removed checkpoint file.")


def main() -> None:
    import os

    api_key = os.environ.get("BYTEZ_API_KEY", "")
    if not api_key:
        logger.error("BYTEZ_API_KEY environment variable not set.")
        sys.exit(1)

    chunks = load_chunks()
    embeddings = embed_with_checkpoint(chunks, api_key)
    build_index(chunks, embeddings)
    logger.info("Done!")


if __name__ == "__main__":
    main()
