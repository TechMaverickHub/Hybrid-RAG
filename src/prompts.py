"""
All prompt templates used across the Hybrid RAG system.
Kept in one file for easy tuning and review.
"""

# ── Query Classifier ──────────────────────────────────────────────
# Used by the router to decide: internal docs, web search, or both.
CLASSIFIER_PROMPT = """\
Classify this question into one category.

Categories:
- internal: answerable from internal documents about {doc_topics}
- external: requires web search for current events, recent news, or general knowledge not in documents
- both: benefits from combining internal documents and web search

Question: {query}

Answer with one word: internal, external, or both."""


# ── Answer Generation ─────────────────────────────────────────────
ANSWER_PROMPT = """\
Answer the following question using the provided context.
The context may come from internal documents, web search results, or both.
Synthesize the information to give a clear, accurate answer.
If the context truly contains no relevant information, say "I don't have that information."
Use three sentences maximum and keep the answer concise.

Context:
{context}

Question:
{question}

Answer:"""


# ── Document Grading ──────────────────────────────────────────────
GRADE_DOCUMENT_PROMPT = """\
Is this document relevant to the question? Answer yes or no.

Document: {context}
Question: {question}

Answer:"""


# ── Query Rewriting ───────────────────────────────────────────────
REWRITE_PROMPT = """\
Rewrite this question to be more specific and search-friendly.
Preserve the original intent.

Original: {question}

Rewritten:"""
