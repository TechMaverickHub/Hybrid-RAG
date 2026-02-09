"""
External tools for the Hybrid RAG system.
Uses LangChain's DuckDuckGoSearchRun for web search (no API key required).
"""

from langchain_community.tools import DuckDuckGoSearchRun


# ── Singleton Search Tool ─────────────────────────────────────────
_search_tool = None


def _get_search_tool() -> DuckDuckGoSearchRun:
    """Load search tool once and reuse."""
    global _search_tool
    if _search_tool is None:
        _search_tool = DuckDuckGoSearchRun(
            num_results=5,
        )
    return _search_tool


def web_search(query: str) -> str:
    """Search the web using DuckDuckGo via LangChain.

    Args:
        query: Search query string

    Returns:
        String of search results with snippets.
    """
    try:
        search = _get_search_tool()
        results = search.run(query)

        print(f"[WEB SEARCH] Query: {query}")
        print(f"[WEB SEARCH] Results length: {len(results) if results else 0}")
        print(f"[WEB SEARCH] Results preview: {results[:500] if results else 'EMPTY'}")

        if not results or results.strip() == "":
            return "No web results found."

        return results

    except Exception as e:
        print(f"[WEB SEARCH] ERROR: {e}")
        return f"Web search error: {str(e)}"
