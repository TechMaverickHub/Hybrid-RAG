"""
Document ingestion pipeline.
Loads PDFs and text files → chunks them → embeds with SentenceTransformer → 
stores in a local FAISS index on disk.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


# ── Document Loading ──────────────────────────────────────────────

def load_documents(data_dir: str) -> list[dict]:
    """Load all .txt and .pdf files from the data directory.
    
    Returns a list of dicts: {"content": str, "source": str, "page": int}
    """
    documents = []

    if not os.path.exists(data_dir):
        print(f"  [WARNING] Data directory '{data_dir}' does not exist.")
        return documents

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)

        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            if text.strip():
                documents.append({
                    "content": text,
                    "source": filename,
                    "page": 0,
                })

        elif filename.endswith(".pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(filepath)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append({
                            "content": text,
                            "source": filename,
                            "page": i + 1,
                        })
            except ImportError:
                print("  [WARNING] 'pypdf' not installed. Skipping PDF files.")
                print("  Run: pip install pypdf")
                break

    return documents


# ── Chunking ──────────────────────────────────────────────────────

def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    
    Each chunk inherits the source metadata from its parent document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )

    chunks = []
    for doc in documents:
        text_chunks = splitter.split_text(doc["content"])
        for chunk_text in text_chunks:
            chunks.append({
                "content": chunk_text,
                "source": doc["source"],
                "page": doc["page"],
            })

    return chunks


# ── FAISS Index Building ─────────────────────────────────────────

def build_faiss_index(chunks: list[dict]) -> tuple:
    """Embed all chunks with SentenceTransformer and build a FAISS index.
    
    Uses IndexFlatL2 — exact L2 distance search (simple but accurate).
    Returns (faiss_index, chunks_list).
    """
    model = SentenceTransformer(settings.embedding_model)

    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    # FAISS expects float32
    embeddings = np.array(embeddings).astype("float32")

    d = embeddings.shape[1]  # vector dimension (384 for all-MiniLM-L6-v2)
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    return index, chunks


# ── Persistence ───────────────────────────────────────────────────

def save_index(index, chunks):
    """Save FAISS index and chunk metadata to disk."""
    os.makedirs(os.path.dirname(settings.index_path), exist_ok=True)
    faiss.write_index(index, settings.index_path)

    with open(settings.chunks_path, "wb") as f:
        pickle.dump(chunks, f)


def load_index():
    """Load FAISS index and chunk metadata from disk."""
    if not os.path.exists(settings.index_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{settings.index_path}'. "
            "Run 'python main.py ingest' first."
        )

    index = faiss.read_index(settings.index_path)

    with open(settings.chunks_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


# ── Full Pipeline ─────────────────────────────────────────────────

def ingest_pipeline():
    """End-to-end ingestion: load → chunk → embed → index → save."""
    print(f"Loading documents from '{settings.data_directory}/'...")
    documents = load_documents(settings.data_directory)
    if not documents:
        print("  No documents found! Place .txt or .pdf files in the data/ folder.")
        return None, None
    print(f"  Loaded {len(documents)} document sections.")

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks.")

    print("Embedding chunks & building FAISS index...")
    index, chunks = build_faiss_index(chunks)
    print(f"  FAISS index: {index.ntotal} vectors, dimension={index.d}")

    print("Saving to disk...")
    save_index(index, chunks)
    print(f"  Index saved to '{settings.index_path}'")
    print(f"  Chunks saved to '{settings.chunks_path}'")

    return index, chunks
