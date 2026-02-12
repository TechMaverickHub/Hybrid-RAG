"""
Pydantic models for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field


# ── Query ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the /api/query endpoint."""
    query: str = Field(..., min_length=1, description="The user's question")


class QueryResponse(BaseModel):
    """Response body from the /api/query endpoint."""
    query: str
    answer: str
    route: str = Field(description="internal | external | both")
    route_reasoning: str
    confidence_met: bool
    sources: list[str] = []


# ── Ingest ────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response body from the /api/ingest endpoint."""
    message: str
    num_documents: int = 0
    num_chunks: int = 0


class UploadResponse(BaseModel):
    """Response body from the /api/upload endpoint."""
    message: str
    filenames: list[str] = []


# ── Health ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body from the /api/health endpoint."""
    status: str = "ok"
    index_loaded: bool = False
    llm_provider: str = ""
    embedding_model: str = ""
