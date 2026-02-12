"""
Query endpoint — the main RAG pipeline.
POST /api/query  →  run the LangGraph workflow and return an answer.
"""

import os
from fastapi import APIRouter, HTTPException

from src.config import settings
from src.schemas import QueryRequest, QueryResponse
from src.graph import build_graph

router = APIRouter()

# ── Lazy-loaded graph singleton ───────────────────────────────────
_graph = None


def _get_graph():
    """Build and cache the LangGraph workflow."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Endpoint ──────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run a query through the Hybrid RAG graph.

    The graph will:
    1. Retrieve from FAISS
    2. Route based on confidence (internal / external / both)
    3. Optionally search the web
    4. Fuse & rerank context
    5. Generate the final answer
    """
    # Check that the FAISS index exists
    if not os.path.exists(settings.index_path):
        raise HTTPException(
            status_code=422,
            detail=(
                "No FAISS index found. "
                "Upload documents and call POST /api/ingest first."
            ),
        )

    graph = _get_graph()

    initial_state = {
        "query": request.query,
        "route": "",
        "route_reasoning": "",
        "doc_results": [],
        "web_results": "",
        "confidence_met": False,
        "fused_context": "",
        "answer": "",
        "sources": [],
    }

    result = graph.invoke(initial_state)

    return QueryResponse(
        query=request.query,
        answer=result["answer"],
        route=result["route"],
        route_reasoning=result["route_reasoning"],
        confidence_met=result["confidence_met"],
        sources=result.get("sources", []),
    )
