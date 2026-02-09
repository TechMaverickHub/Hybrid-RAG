"""
Cross-encoder reranker for context fusion.

Bi-encoder embeddings (FAISS retrieval) are fast but approximate.
The cross-encoder jointly attends to (query, document) pairs for 
more accurate relevance scoring — we only run it on the top-K 
candidates, so it stays fast.
"""

from sentence_transformers import CrossEncoder

from src.config import settings


# ── Singleton Reranker ────────────────────────────────────────────
_reranker = None


def get_reranker() -> CrossEncoder:
    """Load cross-encoder model once and reuse."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


# ── Reranking ─────────────────────────────────────────────────────

def rerank(query: str, documents: list[dict], top_n: int = None) -> list[dict]:
    """Rerank documents using a cross-encoder model.
    
    Args:
        query: User's question
        documents: List of dicts, each must have a 'content' key
        top_n: Number of top documents to keep after reranking
        
    Returns:
        Reranked list of documents (highest relevance first),
        each with an added 'rerank_score' field.
    """
    if not documents:
        return []

    if top_n is None:
        top_n = settings.rerank_top_n

    reranker = get_reranker()

    # Cross-encoder expects (query, passage) pairs
    pairs = [(query, doc["content"]) for doc in documents]
    scores = reranker.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    # Sort by rerank score (highest first) and keep top_n
    reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]
