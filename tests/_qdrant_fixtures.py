"""Shared helper for building a tiny local Qdrant index in tests.

Not a pytest fixture itself (call shapes differ slightly per test module) —
just the index-building logic factored out so test_rag.py and
test_orchestrator.py don't duplicate it.
"""

from pathlib import Path


def build_tmp_qdrant_index(tmp_path: Path, chunks: list[dict]) -> Path:
    """Build a small local-mode Qdrant collection from chunk dicts and return its dir.

    Each chunk dict needs: chunk_id, text, page_start, page_end,
    section_heading, source_url. Uses FakeEmbeddings for the dense vector
    (matches `get_embeddings()`'s offline fallback — no API keys needed) and
    the real (tiny, ~10KB) BM25 sparse model for the sparse vector.
    """
    from fastembed import SparseTextEmbedding
    from langchain_community.embeddings import FakeEmbeddings
    from qdrant_client import QdrantClient, models

    from procurement_copilot.config import settings
    from procurement_copilot.rag.retriever import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

    dense_embedder = FakeEmbeddings(size=1536)
    sparse_embedder = SparseTextEmbedding(model_name=settings.SPARSE_MODEL)

    texts = [c["text"] for c in chunks]
    dense_vecs = dense_embedder.embed_documents(texts)
    sparse_vecs = list(sparse_embedder.embed(texts))

    index_dir = tmp_path / "test_qdrant_index"
    client = QdrantClient(path=str(index_dir))
    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(size=1536, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
    )

    points = [
        models.PointStruct(
            id=i,
            vector={
                DENSE_VECTOR_NAME: dense_vecs[i],
                SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=sparse_vecs[i].indices.tolist(),
                    values=sparse_vecs[i].values.tolist(),
                ),
            },
            payload=chunk,
        )
        for i, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    client.close()

    return index_dir
