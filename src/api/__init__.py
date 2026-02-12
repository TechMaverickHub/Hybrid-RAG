"""
FastAPI route aggregation.
Collects all sub-routers into a single `api_router`.
"""

from fastapi import APIRouter

from src.api.query import router as query_router
from src.api.ingest import router as ingest_router
from src.api.health import router as health_router

api_router = APIRouter(prefix="/api")

api_router.include_router(query_router, tags=["Query"])
api_router.include_router(ingest_router, tags=["Ingest"])
api_router.include_router(health_router, tags=["Health"])
