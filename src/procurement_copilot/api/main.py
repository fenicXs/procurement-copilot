"""FastAPI application entry-point."""

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from procurement_copilot.config import settings
from procurement_copilot.orchestrator.graph import run_query

app = FastAPI(
    title="Procurement Copilot",
    description=(
        "Enterprise agentic AI for procurement policy," " spend analytics, and knowledge graphs"
    ),
    version="0.1.0",
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class QueryResponse(BaseModel):
    question: str
    intent: str
    answer: str
    is_verified: bool
    citations: list[dict] = []
    sql_query: str = ""
    graph_triples: list[dict] = []
    error: str | None = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Run a question through the full orchestrator pipeline."""
    try:
        result = run_query(
            question=req.question,
            index_dir=settings.VECTOR_INDEX_DIR,
            db_path=settings.DB_PATH,
            kg_path=settings.KG_PATH,
        )
        return QueryResponse(**asdict(result))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
