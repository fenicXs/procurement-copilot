"""FastAPI application entry-point."""

import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from procurement_copilot.config import settings
from procurement_copilot.guardrails import check_input
from procurement_copilot.orchestrator.graph import run_query

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"

app = FastAPI(
    title="Procurement Copilot",
    description=(
        "Enterprise agentic AI for procurement policy," " spend analytics, and knowledge graphs"
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Per-IP rate limit on /query — this becomes a public, unauthenticated
# endpoint once deployed, so it needs an abuse/cost guard regardless of
# which LLM provider is paying for tokens behind it.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,  # type: ignore[arg-type]  # slowapi's handler is untyped vs Starlette's generic signature
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
@limiter.limit("20/minute")
async def query(req: QueryRequest, request: Request) -> QueryResponse:
    """Run a question through the full orchestrator pipeline."""
    ok, reason = check_input(req.question)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    try:
        result = run_query(
            question=req.question,
            index_dir=settings.VECTOR_INDEX_DIR,
            db_path=settings.DB_PATH,
            kg_path=settings.KG_PATH,
            session_id=str(uuid.uuid4()),
            use_cache=True,
        )
        return QueryResponse(**asdict(result))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Mounted last so it doesn't shadow the /health and /query routes above —
# Starlette matches routes in registration order, so explicit routes win.
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
