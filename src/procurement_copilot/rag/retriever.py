"""RAG retriever: Qdrant hybrid (dense + sparse) search with cross-encoder reranking.

Retrieval is two-stage:
1. Hybrid search — dense (semantic) + sparse/BM25 (lexical) candidates fused
   via Reciprocal Rank Fusion, so paraphrased questions and exact FAR
   citations ("FAR 6.302-1") are both well served.
2. Rerank — a real cross-encoder (not a heuristic) re-scores the candidate
   pool against the query and the top_k survive.

Replaces the retired FAISS + heading-overlap-heuristic pipeline.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from procurement_copilot.config import settings

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Lazy singletons — the cross-encoder and sparse model are expensive to load
# and shouldn't be reconstructed per query.
_reranker = None
_sparse_embedder = None


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with metadata."""

    chunk_id: str
    text: str
    page_start: int
    page_end: int
    section_heading: str
    source_url: str
    score: float


def _get_embeddings():  # type: ignore[no-untyped-def]
    """Get the best available embedding model (must match what built the index)."""
    from procurement_copilot.llm import get_embeddings

    return get_embeddings()


def _get_sparse_embedder():  # type: ignore[no-untyped-def]
    global _sparse_embedder
    if _sparse_embedder is None:
        from fastembed import SparseTextEmbedding

        _sparse_embedder = SparseTextEmbedding(model_name=settings.SPARSE_MODEL)
    return _sparse_embedder


def _get_reranker():  # type: ignore[no-untyped-def]
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(settings.RERANKER_MODEL)
    return _reranker


def get_qdrant_client(index_dir: Path | None = None):  # type: ignore[no-untyped-def]
    """Open the local-mode Qdrant store (no server process required)."""
    from qdrant_client import QdrantClient

    path = index_dir or settings.VECTOR_INDEX_DIR
    return QdrantClient(path=str(path))


def _hybrid_search(client, collection: str, query: str, limit: int) -> list:  # type: ignore[no-untyped-def]
    """Dense + sparse prefetch fused with Reciprocal Rank Fusion."""
    from qdrant_client import models

    dense_vector = _get_embeddings().embed_query(query)
    sparse_vector = next(_get_sparse_embedder().embed([query]))

    result = client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(query=dense_vector, using=DENSE_VECTOR_NAME, limit=limit),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices.tolist(),
                    values=sparse_vector.values.tolist(),
                ),
                using=SPARSE_VECTOR_NAME,
                limit=limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return list(result.points)


def _rerank(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Re-score candidates with a cross-encoder; higher score = more relevant."""
    if not candidates:
        return candidates

    pairs = [(query, c.text) for c in candidates]
    scores = _get_reranker().predict(pairs)

    for chunk, score in zip(candidates, scores):
        chunk.score = float(score)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def retrieve(
    query: str,
    top_k: int = 5,
    index_dir: Path | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the top-k most relevant chunks for a query.

    Pulls `top_k * 4` hybrid candidates, then reranks with a cross-encoder
    and returns the top_k survivors, best first.
    """
    client = get_qdrant_client(index_dir)
    candidate_limit = top_k * 4

    points = _hybrid_search(client, settings.QDRANT_COLLECTION, query, candidate_limit)

    candidates: list[RetrievedChunk] = []
    for point in points:
        payload = point.payload or {}
        candidates.append(
            RetrievedChunk(
                chunk_id=payload.get("chunk_id", ""),
                text=payload.get("text", ""),
                page_start=payload.get("page_start", 0),
                page_end=payload.get("page_end", 0),
                section_heading=payload.get("section_heading", ""),
                source_url=payload.get("source_url", ""),
                score=float(point.score),
            )
        )

    reranked = _rerank(query, candidates)
    return reranked[:top_k]
