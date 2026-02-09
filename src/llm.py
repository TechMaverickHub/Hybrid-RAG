"""
Unified LLM interface using HuggingFace Inference API.

Supports two providers:
  - "huggingface" → HuggingFace Inference API (HuggingFaceEndpoint + ChatHuggingFace)
  - "groq"        → Groq cloud API with Llama 3.3 70B (free tier, very fast)

Switch via LLM_PROVIDER in .env or config.
"""

from src.config import settings


# ── HuggingFace Model (Lazy-loaded Singleton) ─────────────────────
_hf_llm = None


def _load_hf_model():
    """Load HuggingFace model via Inference API (first call only)."""
    global _hf_llm
    if _hf_llm is None:
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

        endpoint = HuggingFaceEndpoint(
            repo_id=settings.hf_model,
            task="text-generation",
            max_new_tokens=settings.hf_max_new_tokens,
            do_sample=False,
            repetition_penalty=settings.hf_repetition_penalty,
            huggingfacehub_api_token=settings.huggingfacehub_api_token,
        )
        _hf_llm = ChatHuggingFace(llm=endpoint)
    return _hf_llm


# ── Generation Interface ──────────────────────────────────────────

def generate(prompt: str) -> str:
    """Generate text using the configured LLM provider.

    Args:
        prompt: Input prompt text

    Returns:
        Generated text string
    """
    if settings.llm_provider == "groq":
        return _generate_groq(prompt)
    else:
        return _generate_huggingface(prompt)


def _generate_huggingface(prompt: str) -> str:
    """Generate using HuggingFace Inference API."""
    llm = _load_hf_model()
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content


def _generate_groq(prompt: str) -> str:
    """Generate using Groq API (free tier)."""
    from langchain.chat_models import init_chat_model

    llm = init_chat_model(
        f"groq/{settings.groq_model}",
        temperature=0,
        api_key=settings.groq_api_key,
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content
