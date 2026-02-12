"""
Health-check endpoint.
GET /api/health  →  system readiness overview.
"""

import os
from fastapi import APIRouter

from src.config import settings
from src.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Return the current health / readiness of the RAG system."""
    index_exists = os.path.exists(settings.index_path)

    return HealthResponse(
        status="ok",
        index_loaded=index_exists,
        llm_provider=settings.llm_provider,
        embedding_model=settings.embedding_model,
    )
