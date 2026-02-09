"""
FAISS-based retriever with confidence scoring.
Uses SentenceTransformer to embed the query, then searches the FAISS index.
Returns documents with similarity scores and a confidence check.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings


# ── Singleton Embedding Model ─────────────────────────────────────
_embed_model = None


def get_embed_model() -> SentenceTransformer:
    """Load embedding model once and reuse."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(settings.embedding_model)
    return _embed_model


# ── Retrieval ─────────────────────────────────────────────────────

def retrieve(index, chunks: list[dict], query: str, top_k: int = None) -> list[dict]:
    """Search the FAISS index for the top-k most similar chunks.
    
    Args:
        index: FAISS index
        chunks: List of chunk dicts (content, source, page)
        query: User's question
        top_k: Number of results to return
        
    Returns:
        List of chunk dicts with added 'score' field (0-1 similarity).
    """
    if top_k is None:
        top_k = settings.top_k

    model = get_embed_model()
    query_embedding = model.encode([query]).astype("float32")

    # FAISS search returns (distances, indices)
    # IndexFlatL2 returns L2 distances — lower = more similar
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:  # FAISS returns -1 if fewer than k results
            continue

        # Convert L2 distance → 0-1 similarity score
        similarity = 1.0 / (1.0 + float(dist))

        chunk = chunks[idx].copy()
        chunk["score"] = similarity
        results.append(chunk)

    return results


# ── Confidence Check ──────────────────────────────────────────────

def check_confidence(results: list[dict], threshold: float = None) -> bool:
    """Check if the top retrieval result meets the confidence threshold.
    
    If the best document's similarity score is below the threshold,
    the system should fall back to web search.
    """
    if threshold is None:
        threshold = settings.similarity_threshold

    if not results:
        return False

    top_score = results[0]["score"]
    return top_score >= threshold
