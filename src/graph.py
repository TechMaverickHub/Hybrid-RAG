"""
LangGraph workflow — the core routing graph.

This is the HEART of the Hybrid RAG system. It uses a
RETRIEVE-FIRST architecture:

  1. Always retrieve from FAISS first (fast)
  2. Use the actual similarity scores to decide routing:
     - HIGH confidence (>= 0.50) → docs only (fast path)
     - MEDIUM confidence (0.35–0.50) → docs + web search (hybrid)
     - LOW confidence (< 0.35) → web search only (fallback)
  3. Fuse and rerank all context with a cross-encoder
  4. Generate the final answer

This is more reliable than LLM-based classification because
it uses real retrieval signals, not guesses.

Graph structure:

  START → retrieve_docs → [confidence routing]
             ├── HIGH   → fuse_context ──────────────────→ generate_answer → END
             ├── MEDIUM → also_search_web → fuse_context → generate_answer → END
             └── LOW    → search_web_only → fuse_context → generate_answer → END
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from src.config import settings
from src.retriever import retrieve
from src.reranker import rerank
from src.ingest import load_index
from src.tools import web_search
from src.llm import generate
from src.prompts import ANSWER_PROMPT


# ── Graph State ───────────────────────────────────────────────────

class GraphState(TypedDict):
    """State that flows through every node in the graph."""
    query: str               # Original user query
    route: str               # "internal" | "external" | "both"
    route_reasoning: str     # Why this route was chosen
    doc_results: list[dict]  # Retrieved document chunks with scores
    web_results: str         # Web search results (formatted text)
    confidence_met: bool     # Did doc retrieval meet the threshold?
    fused_context: str       # Merged + reranked context for the LLM
    answer: str              # Final generated answer
    sources: list[str]       # Source attribution list


# ── Node Functions ────────────────────────────────────────────────

def retrieve_docs_node(state: GraphState) -> dict:
    """Node 1: Always retrieve from FAISS first.
    
    This is fast and gives us real similarity scores
    to make intelligent routing decisions.
    """
    index, chunks = load_index()
    results = retrieve(index, chunks, state["query"])

    # Determine confidence level from the top result's score
    top_score = results[0]["score"] if results else 0.0
    confidence = top_score >= settings.similarity_threshold

    return {
        "doc_results": results,
        "confidence_met": confidence,
    }


def also_search_web_node(state: GraphState) -> dict:
    """Node 2b: MEDIUM confidence — also search the web to supplement docs."""
    results = web_search(state["query"])
    return {"web_results": results}


def search_web_only_node(state: GraphState) -> dict:
    """Node 2c: LOW confidence — docs aren't useful, rely on web search."""
    results = web_search(state["query"])
    return {
        "web_results": results,
        "doc_results": [],  # discard low-quality doc results
    }


def fuse_context_node(state: GraphState) -> dict:
    """Node 3: Merge document + web results, then rerank with cross-encoder."""
    all_docs = []
    sources = []

    # Add document results
    for doc in state.get("doc_results") or []:
        all_docs.append(doc)
        sources.append(
            f"[DOC] {doc['source']} (page {doc['page']}, score={doc['score']:.3f})"
        )

    # Add web results
    web = state.get("web_results") or ""
    if web and web not in ("No web results found.", ""):
        all_docs.append({
            "content": web,
            "source": "web_search",
            "page": 0,
            "score": 0.5,  # neutral score for web results
        })
        sources.append("[WEB] DuckDuckGo Search")

    # Rerank with cross-encoder if we have multiple sources
    if len(all_docs) > 1:
        reranked = rerank(state["query"], all_docs)
    else:
        reranked = all_docs

    # Build the fused context string
    fused = "\n\n---\n\n".join([doc["content"] for doc in reranked])

    return {
        "fused_context": fused,
        "sources": sources,
    }


def generate_answer_node(state: GraphState) -> dict:
    """Node 4: Generate the final answer using the LLM."""
    prompt = ANSWER_PROMPT.format(
        question=state["query"],
        context=state["fused_context"],
    )
    answer = generate(prompt)
    return {"answer": answer}


# ── Conditional Edge: Confidence-Based Routing ────────────────────

def confidence_routing(state: GraphState) -> str:
    """Route based on actual FAISS retrieval scores.

    This replaces the unreliable LLM classifier with real data:
      - HIGH  (>= 0.50): Docs clearly answer the question → skip web
      - MEDIUM (0.35–0.50): Docs might help, but let's also check web
      - LOW   (< 0.35): Docs are irrelevant → web search only
    """
    doc_results = state.get("doc_results") or []
    top_score = doc_results[0]["score"] if doc_results else 0.0

    if top_score >= settings.high_confidence_threshold:
        # HIGH — docs are sufficient, no need for web search
        return "fuse_context"

    elif top_score >= settings.similarity_threshold:
        # MEDIUM — docs have some relevance, supplement with web
        return "also_search_web"

    else:
        # LOW — docs aren't helpful, go straight to web
        return "search_web_only"


def set_route_internal(state: GraphState) -> dict:
    """Set route metadata for HIGH confidence path (docs only)."""
    top_score = state["doc_results"][0]["score"] if state.get("doc_results") else 0.0
    return {
        "route": "internal",
        "route_reasoning": f"High doc confidence (top_score={top_score:.3f} >= {settings.high_confidence_threshold}). Docs sufficient.",
    }


def set_route_both(state: GraphState) -> dict:
    """Set route metadata for MEDIUM confidence path (docs + web)."""
    top_score = state["doc_results"][0]["score"] if state.get("doc_results") else 0.0
    return {
        "route": "both",
        "route_reasoning": f"Medium doc confidence (top_score={top_score:.3f}). Supplementing with web search.",
    }


def set_route_external(state: GraphState) -> dict:
    """Set route metadata for LOW confidence path (web only)."""
    top_score = state["doc_results"][0]["score"] if state.get("doc_results") else 0.0
    return {
        "route": "external",
        "route_reasoning": f"Low doc confidence (top_score={top_score:.3f} < {settings.similarity_threshold}). Using web search.",
    }


# ── Graph Assembly ────────────────────────────────────────────────

def build_graph():
    """Build and compile the LangGraph workflow.

    Architecture: Retrieve-First Routing
    
    Instead of guessing with an LLM classifier, we:
    1. Always retrieve from FAISS first (fast, ~10ms)
    2. Use actual similarity scores to route intelligently
    3. Only call web search when docs aren't confident enough

    Returns a compiled graph that can be invoked with:
        result = graph.invoke({"query": "...", ...})
    """
    workflow = StateGraph(GraphState)

    # ── Add Nodes ──
    workflow.add_node("retrieve_docs", retrieve_docs_node)
    workflow.add_node("set_route_internal", set_route_internal)
    workflow.add_node("set_route_both", set_route_both)
    workflow.add_node("set_route_external", set_route_external)
    workflow.add_node("also_search_web", also_search_web_node)
    workflow.add_node("search_web_only", search_web_only_node)
    workflow.add_node("fuse_context", fuse_context_node)
    workflow.add_node("generate_answer", generate_answer_node)

    # ── Edges ──

    # START → always retrieve from docs first
    workflow.add_edge(START, "retrieve_docs")

    # After retrieval → route based on confidence scores
    workflow.add_conditional_edges(
        "retrieve_docs",
        confidence_routing,
        {
            "fuse_context": "set_route_internal",         # HIGH → docs only
            "also_search_web": "set_route_both",          # MEDIUM → docs + web
            "search_web_only": "set_route_external",      # LOW → web only
        },
    )

    # HIGH path: set metadata → fuse (docs only) → generate
    workflow.add_edge("set_route_internal", "fuse_context")

    # MEDIUM path: set metadata → web search → fuse (docs + web) → generate
    workflow.add_edge("set_route_both", "also_search_web")
    workflow.add_edge("also_search_web", "fuse_context")

    # LOW path: set metadata → web search → fuse (web only) → generate
    workflow.add_edge("set_route_external", "search_web_only")
    workflow.add_edge("search_web_only", "fuse_context")

    # All paths converge → fuse → generate → END
    workflow.add_edge("fuse_context", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()
