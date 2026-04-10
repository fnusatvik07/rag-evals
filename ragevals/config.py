"""Configuration management for RAG pipelines."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


def load_env(dotenv_path: str | None = None) -> dict:
    """Load environment variables from .env file.

    Searches upward from CWD if no path is provided.
    Raises EnvironmentError if OPENAI_API_KEY is not set.

    Returns:
        Dict of loaded environment variable names and values.
    """
    if dotenv_path and os.path.exists(dotenv_path):
        load_dotenv(dotenv_path, override=True)
    else:
        # Walk up from CWD to find .env
        current = Path.cwd()
        found = False
        for parent in [current] + list(current.parents):
            candidate = parent / ".env"
            if candidate.exists():
                load_dotenv(str(candidate), override=True)
                found = True
                break
        if not found:
            load_dotenv(override=True)  # Try default locations

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file or export it."
        )

    loaded = {}
    for key in ["OPENAI_API_KEY", "CONFIDENT_API_KEY"]:
        val = os.getenv(key)
        if val:
            loaded[key] = val

    return loaded


@dataclass
class RAGConfig:
    """Configuration for a RAG pipeline."""

    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    system_prompt: str = field(default=(
        "You are a helpful customer support assistant for Acme Corp. "
        "Answer the customer's question based ONLY on the provided context documents. "
        "If the context does not contain enough information to answer the question, say so clearly. "
        "Do not make up information that is not supported by the context. "
        "Be concise but thorough in your response."
    ))
    collection_name: str = "acme_corp_kb"
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @property
    def name(self) -> str:
        """Short name for this configuration."""
        return f"chunk{self.chunk_size}_top{self.top_k}_temp{self.temperature}"

    @classmethod
    def from_dict(cls, d: dict) -> "RAGConfig":
        """Create a RAGConfig from a dictionary (e.g., parsed YAML)."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
