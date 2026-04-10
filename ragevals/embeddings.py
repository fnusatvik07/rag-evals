"""Embedding utilities using OpenAI."""

from __future__ import annotations


def get_embeddings(
    texts: list[str],
    model: str = "text-embedding-3-small",
    client=None,
) -> list[list[float]]:
    """Get embeddings for a list of texts using OpenAI.

    Args:
        texts: List of text strings to embed.
        model: OpenAI embedding model name.
        client: Optional pre-initialized OpenAI client.

    Returns:
        List of embedding vectors (list of floats).
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI()

    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]
