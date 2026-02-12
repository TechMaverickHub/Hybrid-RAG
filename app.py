"""
Streamlit UI for the Hybrid RAG system.
Communicates with the FastAPI backend via HTTP.

Run with:
    1. Start the backend:  uvicorn main:app --reload
    2. Start the frontend: streamlit run app.py
"""

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api"


# ── Helpers ───────────────────────────────────────────────────────

def api_health() -> dict | None:
    """Check backend health."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return None
    except Exception:
        return None


def api_query(query: str) -> dict:
    """Send a query to the RAG backend."""
    resp = requests.post(f"{API_BASE}/query", json={"query": query}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def api_upload(files) -> dict:
    """Upload files to the backend."""
    multipart = [("files", (f.name, f.read(), f.type)) for f in files]
    resp = requests.post(f"{API_BASE}/upload", files=multipart, timeout=60)
    resp.raise_for_status()
    return resp.json()


def api_ingest() -> dict:
    """Trigger the ingestion pipeline on the backend."""
    resp = requests.post(f"{API_BASE}/ingest", timeout=300)
    resp.raise_for_status()
    return resp.json()


# ── Page Config ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="🔍",
    layout="wide",
)


# ── Sidebar: Health / Upload / Settings ───────────────────────────

with st.sidebar:
    st.title("Hybrid RAG")
    st.caption("Docs + Web Search")

    st.divider()

    # --- Backend Status ---
    health = api_health()
    if health is None:
        st.error(
            "Backend offline. Start it with:\n\n"
            "```\nuvicorn main:app --reload\n```"
        )
    else:
        st.success("Backend connected")
        if health.get("index_loaded"):
            st.success("FAISS index loaded")
        else:
            st.warning("No index found. Upload and ingest documents first.")

    st.divider()

    # --- Document Upload ---
    st.header("Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Upload & Ingest", type="primary"):
        with st.spinner("Uploading files..."):
            try:
                upload_resp = api_upload(uploaded_files)
                st.info(upload_resp["message"])
            except Exception as e:
                st.error(f"Upload failed: {e}")
                st.stop()

        with st.spinner("Ingesting documents (embedding + indexing)..."):
            try:
                ingest_resp = api_ingest()
                st.success(
                    f"{ingest_resp['message']}  \n"
                    f"Documents: {ingest_resp['num_documents']} · "
                    f"Chunks: {ingest_resp['num_chunks']}"
                )
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()

    # --- Settings (read-only, from backend) ---
    st.header("Settings")
    if health:
        st.text(f"LLM Provider: {health.get('llm_provider', '—')}")
        st.text(f"Embedding: {health.get('embedding_model', '—')}")


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

    # Guard: backend must be running
    if health is None:
        with st.chat_message("assistant"):
            st.warning("Backend is offline. Start it first (see sidebar).")
    elif not health.get("index_loaded"):
        with st.chat_message("assistant"):
            st.warning(
                "No documents ingested yet! Upload files in the sidebar "
                "and click 'Upload & Ingest' first."
            )
    else:
        # Call the FastAPI backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = api_query(query)
                except requests.HTTPError as e:
                    st.error(f"Query failed: {e.response.text}")
                    st.stop()
                except requests.ConnectionError:
                    st.error("Lost connection to backend.")
                    st.stop()

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
