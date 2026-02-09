"""
Streamlit UI for the Hybrid RAG system.

Run with:
    streamlit run app.py
"""

import os
import sys
import streamlit as st

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import settings


# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="🔍",
    layout="wide",
)


# ── Sidebar: Document Ingestion & Settings ────────────────────────
with st.sidebar:
    st.title("Hybrid RAG")
    st.caption("Docs + Web Search")

    st.divider()

    # --- Document Upload ---
    st.header("Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest Documents", type="primary"):
        os.makedirs(settings.data_directory, exist_ok=True)

        for f in uploaded_files:
            filepath = os.path.join(settings.data_directory, f.name)
            with open(filepath, "wb") as out:
                out.write(f.read())

        with st.spinner("Ingesting documents..."):
            from src.ingest import ingest_pipeline
            ingest_pipeline()

        st.success(f"Ingested {len(uploaded_files)} file(s)!")

    st.divider()

    # --- Settings ---
    st.header("Settings")

    st.text(f"LLM Provider: {settings.llm_provider}")
    st.text(f"Embedding: {settings.embedding_model}")

    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=settings.similarity_threshold,
        step=0.05,
        help="Below this threshold, the system falls back to web search.",
    )
    # Update threshold dynamically
    settings.similarity_threshold = threshold

    st.divider()

    # --- Index Status ---
    if os.path.exists(settings.index_path):
        st.success("FAISS index loaded")
    else:
        st.warning("No index found. Upload and ingest documents first.")


# ── Main Chat Area ────────────────────────────────────────────────
st.title("Hybrid RAG — Ask Anything")
st.markdown(
    "This system dynamically routes your query to **internal documents**, "
    "**web search**, or **both** — then fuses and reranks the results."
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("metadata"):
            with st.expander("Routing Details"):
                meta = msg["metadata"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Route", meta.get("route", "—"))
                with col2:
                    st.metric("Confidence", "MET" if meta.get("confidence_met") else "LOW")
                st.write(f"**Reasoning:** {meta.get('route_reasoning', '—')}")
                if meta.get("sources"):
                    st.write("**Sources:**")
                    for s in meta["sources"]:
                        st.write(f"- {s}")


# Chat input
query = st.chat_input("Ask a question...")

if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Check index exists
    if not os.path.exists(settings.index_path):
        with st.chat_message("assistant"):
            st.warning(
                "No documents ingested yet! Upload files in the sidebar "
                "and click 'Ingest Documents' first."
            )
    else:
        # Build graph and invoke
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                from src.graph import build_graph

                graph = build_graph()
                result = graph.invoke({
                    "query": query,
                    "route": "",
                    "route_reasoning": "",
                    "doc_results": [],
                    "web_results": "",
                    "confidence_met": False,
                    "fused_context": "",
                    "answer": "",
                    "sources": [],
                })

            # Display answer
            st.write(result["answer"])

            # Show routing details
            with st.expander("Routing Details"):
                col1, col2 = st.columns(2)
                with col1:
                    route_color = {
                        "internal": "🟢",
                        "external": "🌐",
                        "both": "🔀",
                    }
                    st.metric(
                        "Route",
                        f"{route_color.get(result['route'], '')} {result['route']}",
                    )
                with col2:
                    st.metric(
                        "Confidence",
                        "MET" if result["confidence_met"] else "LOW (fallback)",
                    )
                st.write(f"**Reasoning:** {result['route_reasoning']}")

                if result.get("sources"):
                    st.write("**Sources:**")
                    for s in result["sources"]:
                        st.write(f"- {s}")

        # Save to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "metadata": {
                "route": result["route"],
                "route_reasoning": result["route_reasoning"],
                "confidence_met": result["confidence_met"],
                "sources": result.get("sources", []),
            },
        })
