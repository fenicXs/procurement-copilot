"""RAG retriever: Qdrant hybrid (dense + sparse) search with cross-encoder reranking.

Retrieval is two-stage:
1. Hybrid search — dense (semantic) + sparse/BM25 (lexical) candidates fused
   via Reciprocal Rank Fusion, so paraphrased questions and exact FAR
   citations ("FAR 6.302-1") are both well served.
2. Rerank — a real cross-encoder (not a heuristic) re-scores the candidate
   pool against the query and the top_k survive.

Replaces the retired FAISS + heading-overlap-heuristic pipeline.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from procurement_copilot.config import settings

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Lazy singletons — the cross-encoder and sparse model are expensive to load
# and shouldn't be reconstructed per query.
_reranker: Any = None
_sparse_embedder = None


def _resolve_embedding_provider() -> str:
    """Same precedence as llm.get_embeddings() — kept in sync deliberately so
    the reranker backend follows the same provider knob as the embedder."""
    return (
        os.environ.get("EMBEDDING_PROVIDER", "")
        or settings.EMBEDDING_PROVIDER
        or os.environ.get("LLM_PROVIDER", "")
        or settings.LLM_PROVIDER
    ).lower()


class _FastEmbedRerankerAdapter:
    """Adapts fastembed's TextCrossEncoder.rerank(query, docs) to the
    CrossEncoder.predict(pairs) shape _rerank() already calls — ONNX-only,
    no torch/sentence-transformers. Used when EMBEDDING_PROVIDER=fastembed:
    that torch dependency chain is what OOM-killed the first free-tier
    (512MB RAM) cloud deploy attempt."""

    def __init__(self, model_name: str, threads: int | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name, threads=threads or len(os.sched_getaffinity(0)))

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        query = pairs[0][0]
        docs = [doc for _, doc in pairs]
        return list(self._model.rerank(query, docs))


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


def _get_reranker() -> Any:
    global _reranker
    if _reranker is None:
        if _resolve_embedding_provider() == "fastembed":
            _reranker = _FastEmbedRerankerAdapter(settings.FASTEMBED_RERANKER_MODEL)
        else:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(settings.RERANKER_MODEL)
    return _reranker


_UNSET: Any = object()
_reranker2: Any = _UNSET


def _get_reranker2() -> Any:
    """Stage-2 reranker, or None when disabled / not on the fastembed path /
    the model can't be loaded (loading is attempted once, then cached)."""
    global _reranker2
    if _reranker2 is _UNSET:
        _reranker2 = None
        if settings.FASTEMBED_RERANKER2_MODEL and _resolve_embedding_provider() == "fastembed":
            try:
                _reranker2 = _FastEmbedRerankerAdapter(settings.FASTEMBED_RERANKER2_MODEL)
            except Exception:
                logger.warning("Stage-2 reranker unavailable — using stage 1 only.", exc_info=True)
    return _reranker2


def _rerank_stage2(query: str, ranked: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Re-score the top `RERANK_SHORTLIST` of an already-reranked list with the
    stronger model; the tail keeps its stage-1 order. No-op if unavailable."""
    reranker = _get_reranker2()
    shortlist = max(settings.RERANK_SHORTLIST, 0)
    if reranker is None or shortlist == 0 or not ranked:
        return ranked

    head, tail = ranked[:shortlist], ranked[shortlist:]
    try:
        scores = reranker.predict([(query, c.text) for c in head])
    except Exception:
        logger.warning("Stage-2 rerank failed — keeping stage-1 order.", exc_info=True)
        return ranked

    for chunk, score in zip(head, scores):
        chunk.score = float(score)
    head.sort(key=lambda c: c.score, reverse=True)
    return head + tail


# --- Section-number pinning -------------------------------------------------
# A question like "What does FAR 6.302-1 say...?" names the section it wants,
# but semantic rerankers rank chunks that merely *cite* "6.302-1" above the
# chunk that *is* that section (it sat at rank 19). Pinning is a pure text
# lookup — no model, negligible memory — so it's safe on the small instance.

_QUERY_SECTION_RE = re.compile(r"(?<![\w.])(\d{1,2}\.\d{3}(?:-\d+)?)(?![\w-]|\.\d)")
_HEADING_RE = re.compile(r"(?<![\w.(\-])(\d{1,2}\.\d{3}(?:-\d+)?)(?![\w\-]|\.\d)\s+(?=[A-Z])")
_NEXT_SECTION_RE = re.compile(r"(?<![\w.(\-])\d{1,2}\.\d{3}(?:-\d+)?\s+[A-Z]")
_SUBPART_RE = re.compile(r"\bSubpart\s+\d+\.\d+\b")
MAX_PINNED = 2

_section_index: dict[str, list[RetrievedChunk]] | None = None


def _is_section_body(text: str, match: re.Match[str]) -> bool:
    """True when the heading hit starts real section text — not a table-of-contents
    line (another section heading follows immediately), a page running header
    ("6.302-1 Subpart 6.3 - ..."), or a Subpart banner."""
    tail = text[match.end() : match.end() + 260]
    return not (
        tail.startswith("Subpart") or _NEXT_SECTION_RE.search(tail) or _SUBPART_RE.search(tail)
    )


def _load_section_index() -> dict[str, list[RetrievedChunk]]:
    path = Path(settings.DATA_DIR) / "processed" / "far_chunks.jsonl"
    index: dict[str, list[RetrievedChunk]] = {}
    if not path.is_file():
        return index
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            text = rec.get("text", "")
            seen: set[str] = set()
            for m in _HEADING_RE.finditer(text):
                if m.group(1) not in seen and _is_section_body(text, m):
                    seen.add(m.group(1))
                    index.setdefault(m.group(1), []).append(
                        RetrievedChunk(
                            chunk_id=rec.get("chunk_id", ""),
                            text=text,
                            page_start=rec.get("page_start", 0),
                            page_end=rec.get("page_end", 0),
                            section_heading=rec.get("section_heading", ""),
                            source_url=rec.get("source_url", ""),
                            score=0.0,
                        )
                    )
    return index


def _pinned_chunks(query: str) -> list[RetrievedChunk]:
    """Chunks that open the FAR section(s) the query names, at most MAX_PINNED."""
    global _section_index
    sections = list(dict.fromkeys(_QUERY_SECTION_RE.findall(query)))
    if not sections:
        return []
    if _section_index is None:
        _section_index = _load_section_index()
    pinned: list[RetrievedChunk] = []
    for sec in sections:
        pinned.extend(_section_index.get(sec, [])[:1])
    return pinned[:MAX_PINNED]


def _apply_pins(query: str, ranked: list[RetrievedChunk]) -> list[RetrievedChunk]:
    pinned = _pinned_chunks(query)
    if not pinned:
        return ranked
    pinned_ids = {c.chunk_id for c in pinned}
    return pinned + [c for c in ranked if c.chunk_id not in pinned_ids]


_qdrant_clients: dict[str, Any] = {}


def get_qdrant_client(index_dir: Path | None = None):  # type: ignore[no-untyped-def]
    """Open the local-mode Qdrant store (no server process required).

    One client per index path, reused across queries: local mode loads the
    whole collection into RAM, so opening (and never closing) a fresh client
    per query leaked a full index copy each time and OOM-killed the 2GiB
    Cloud Run instance on the second request.
    """
    from qdrant_client import QdrantClient

    path = str(index_dir or settings.VECTOR_INDEX_DIR)
    if path not in _qdrant_clients:
        _qdrant_clients[path] = QdrantClient(path=path)
    return _qdrant_clients[path]


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


MIN_CANDIDATE_POOL = 100


def retrieve(
    query: str,
    top_k: int = 5,
    index_dir: Path | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the top-k most relevant chunks for a query.

    Pulls a wide hybrid candidate pool, then reranks with a cross-encoder
    and returns the top_k survivors, best first.

    The pool floor (`MIN_CANDIDATE_POOL`) matters more than it looks: FAR is
    full of chunks that merely *reference* a defined term (e.g. "the
    simplified acquisition threshold") without stating its value, so the one
    chunk that actually defines it can rank well outside `top_k * 4` on both
    dense and sparse signals alone — confirmed empirically, it was sitting at
    hybrid rank ~58 for a flagship demo question. The cross-encoder reranker
    disambiguates this correctly once it's given a wide enough pool to see
    the chunk at all (promoted it to rank 1 at pool size 100).
    """
    client = get_qdrant_client(index_dir)
    candidate_limit = max(top_k * 4, MIN_CANDIDATE_POOL)

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

    reranked = _rerank_stage2(query, _rerank(query, candidates))
    # Only the production corpus has the section index; a caller-supplied
    # index (tests, fixtures) must not get real FAR chunks injected.
    if index_dir is None or Path(index_dir) == Path(settings.VECTOR_INDEX_DIR):
        reranked = _apply_pins(query, reranked)
    return reranked[:top_k]
