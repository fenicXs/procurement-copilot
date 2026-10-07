"""Build a Qdrant hybrid (dense + sparse) index from FAR chunks.

Usage:
    python scripts/build_qdrant_index.py [--input data/processed/far_chunks.jsonl]
                                         [--output-dir data/processed/vector_index]

Uses Qdrant's embedded/local mode (QdrantClient(path=...)) — no server
process required, so this works on HPC nodes without Docker. Each point
carries a named dense vector (from the configured embedding provider) and a
named sparse vector (BM25 via fastembed), enabling hybrid search with RRF
fusion at query time. Replaces the retired FAISS-based build scripts.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "far_chunks.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "vector_index"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
BATCH_SIZE = 64

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    logger.info("Loaded %d chunks from %s.", len(chunks), path)
    return chunks


def build_qdrant_index(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    collection_name: str | None = None,
) -> Path:
    """Build a Qdrant local-mode hybrid index from chunks and persist to disk."""
    from fastembed import SparseTextEmbedding
    from qdrant_client import QdrantClient, models

    from procurement_copilot.config import settings
    from procurement_copilot.llm import get_embeddings

    collection = collection_name or settings.QDRANT_COLLECTION
    chunks = _load_chunks(input_path)
    texts = [c["text"] for c in chunks]

    dense_embedder = get_embeddings()
    sparse_embedder = SparseTextEmbedding(model_name=settings.SPARSE_MODEL)

    # Dense dimensionality varies by provider (768 for nomic-embed-text via
    # Ollama, 1536 for OpenAI text-embedding-3-small) — probe it rather than
    # hardcoding.
    probe_vector = dense_embedder.embed_query(texts[0])
    dense_dim = len(probe_vector)
    logger.info("Dense embedding dimension: %d", dense_dim)

    output_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(output_dir))

    if client.collection_exists(collection):
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
    )

    logger.info("Embedding and upserting %d chunks in batches of %d...", len(chunks), BATCH_SIZE)
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        batch_texts = texts[start : start + BATCH_SIZE]

        dense_vecs = dense_embedder.embed_documents(batch_texts)
        sparse_vecs = list(sparse_embedder.embed(batch_texts))

        points = []
        for i, chunk in enumerate(batch):
            points.append(
                models.PointStruct(
                    id=start + i,
                    vector={
                        DENSE_VECTOR_NAME: dense_vecs[i],
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=sparse_vecs[i].indices.tolist(),
                            values=sparse_vecs[i].values.tolist(),
                        ),
                    },
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                        "section_heading": chunk.get("section_heading", ""),
                        "source_url": chunk["source_url"],
                    },
                )
            )

        client.upsert(collection_name=collection, points=points)
        if (start // BATCH_SIZE) % 10 == 0:
            logger.info("Upserted %d / %d chunks.", start + len(batch), len(chunks))

    logger.info("Built Qdrant collection '%s' at %s.", collection, output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qdrant hybrid index from FAR chunks")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    build_qdrant_index(input_path=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
    sys.exit(0)
