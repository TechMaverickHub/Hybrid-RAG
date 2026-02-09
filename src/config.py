"""
Central configuration for the Hybrid RAG system.
Uses pydantic-settings to load from .env file and environment variables.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM Provider ──────────────────────────────────────────────
    # "huggingface" = HuggingFace Inference API (default)
    # "groq"        = Groq cloud API (free tier, very fast)
    llm_provider: str = "huggingface"

    # HuggingFace settings (used when llm_provider == "huggingface")
    huggingfacehub_api_token: str = ""
    hf_model: str = "openai/gpt-oss-20b"
    hf_max_new_tokens: int = 512
    hf_repetition_penalty: float = 1.03

    # Groq settings (used when llm_provider == "groq")
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # ── Embedding Model ───────────────────────────────────────────
    # Runs locally via SentenceTransformer — no API key needed
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Chunking ──────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── Retrieval ─────────────────────────────────────────────────
    top_k: int = 5
    similarity_threshold: float = 0.35   # below this → web search only
    high_confidence_threshold: float = 0.50  # above this → docs only (skip web)

    # ── Reranker ──────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 3

    # ── Paths ─────────────────────────────────────────────────────
    data_directory: str = "data"
    index_path: str = "vectorstore/index.faiss"
    chunks_path: str = "vectorstore/chunks.pkl"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
