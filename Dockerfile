FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

# Editable install on purpose: config.py and api/main.py resolve data/ and
# frontend/ relative to __file__, which must stay under /app/src (a regular
# install moves the package into site-packages and breaks those paths).
RUN pip install --no-cache-dir -e .

# Bake the fastembed models (dense, BM25 sparse, reranker) into the image.
# Downloading them lazily at runtime failed on Cloud Run ("Could not load
# model ... from any source") and would also add a download to every cold start.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed_cache
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); \
SparseTextEmbedding('Qdrant/bm25'); \
TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2'); \
TextCrossEncoder('BAAI/bge-reranker-v2-m3-int8')"

COPY frontend/ frontend/
COPY data/processed/far_chunks.jsonl data/processed/far_chunks.jsonl
COPY data/processed/procurement.duckdb data/processed/procurement.duckdb
COPY data/processed/kg.parquet data/processed/kg.parquet
COPY data/processed/vector_index_cloud/ data/processed/vector_index/

# Public demo defaults — generation via Groq's free tier, retrieval via
# fastembed (CPU-only, no API key). Override GROQ_API_KEY as a secret in
# whatever platform deploys this image (e.g. HuggingFace Space secrets).
# RERANKER_MODEL is overridden to a much lighter cross-encoder than the
# SOL/GPU default (BAAI/bge-reranker-v2-m3, ~568M params) — on 2-4 shared
# CPUs that model takes 100+ seconds to rerank a 100-candidate pool, which
# is unusable for an interactive demo. ms-marco-MiniLM-L-6-v2 (~22M params)
# reranks the same pool in ~4-5s with no loss in top-1 quality (verified).
ENV LLM_PROVIDER=groq
ENV EMBEDDING_PROVIDER=fastembed
ENV RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

EXPOSE 8000

CMD ["uvicorn", "procurement_copilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
