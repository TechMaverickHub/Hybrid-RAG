"""
FastAPI entry point for the Hybrid RAG system.

Run with:
    uvicorn main:app --reload
    # or
    python main.py
"""

import sys
import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api import api_router
from src.config import settings


# ── Lifespan (startup / shutdown) ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load heavy models at startup so the first request is fast."""
    print("=" * 60)
    print("  HYBRID RAG — FastAPI Backend")
    print(f"  LLM Provider : {settings.llm_provider}")
    print(f"  Embedding    : {settings.embedding_model}")
    print(f"  Index path   : {settings.index_path}")
    print("=" * 60)

    # Pre-load embedding model (used by retriever + ingest)
    try:
        from src.retriever import get_embed_model
        print("Loading embedding model...")
        get_embed_model()
        print("  Embedding model ready.")
    except Exception as e:
        print(f"  [WARNING] Could not pre-load embedding model: {e}")

    # Pre-load reranker
    try:
        from src.reranker import get_reranker
        print("Loading reranker model...")
        get_reranker()
        print("  Reranker model ready.")
    except Exception as e:
        print(f"  [WARNING] Could not pre-load reranker: {e}")

    if os.path.exists(settings.index_path):
        print("FAISS index found — ready for queries.")
    else:
        print("No FAISS index found — upload & ingest documents first.")

    print("-" * 60)
    yield
    print("Shutting down Hybrid RAG.")


# ── FastAPI App ───────────────────────────────────────────────────

app = FastAPI(
    title="Hybrid RAG API",
    description=(
        "A Retrieval-Augmented Generation system that dynamically routes "
        "queries to internal documents, web search, or both — then fuses "
        "and reranks the results."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Streamlit frontend (and any other local client)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all API routes under /api
app.include_router(api_router)


# ── Run with `python main.py` ────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
