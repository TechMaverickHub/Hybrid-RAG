"""
Query intent classifier (the Router).

NOTE: The main graph uses RETRIEVE-FIRST routing (confidence-based),
which is more reliable than LLM classification. This classifier is
still available as a utility — useful for:
  - Logging/analytics (classify queries for reporting)
  - Groq mode (where the LLM is smart enough to classify accurately)
  - Future use as a secondary routing signal
"""

from src.llm import generate
from src.prompts import CLASSIFIER_PROMPT


def classify_query(query: str, doc_topics: str = "uploaded documents") -> dict:
    """Classify query intent into: internal, external, or both.
    
    Uses the LLM to analyze the question and route it to the
    appropriate knowledge source.
    
    Args:
        query: User's question
        doc_topics: Description of what the internal documents cover
        
    Returns:
        {"route": "internal"|"external"|"both", "reasoning": str}
    """
    prompt = CLASSIFIER_PROMPT.format(query=query, doc_topics=doc_topics)
    response = generate(prompt).strip().lower()

    # Parse the LLM response — works for both flan-t5 and Groq
    if "external" in response:
        route = "external"
    elif "both" in response:
        route = "both"
    elif "internal" in response:
        route = "internal"
    else:
        # Fallback: if the model output is unclear, default to "both"
        route = "both"

    return {"route": route, "reasoning": response}
