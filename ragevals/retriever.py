"""Retrieval and reranking functions."""

from __future__ import annotations

import os
from .embeddings import get_embeddings


def retrieve(
    query: str,
    client,
    collection_name: str,
    top_k: int = 3,
    embedding_model: str = "text-embedding-3-small",
    openai_client=None,
) -> list[dict]:
    """Retrieve top-k relevant chunks from Qdrant.

    Args:
        query: Search query.
        client: QdrantClient instance.
        collection_name: Name of the Qdrant collection.
        top_k: Number of results to return.
        embedding_model: Model for query embedding.
        openai_client: Optional OpenAI client.

    Returns:
        List of dicts with keys: text, title, score.
    """
    query_embedding = get_embeddings([query], model=embedding_model, client=openai_client)[0]

    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    retrieved = []
    for point in results.points:
        retrieved.append({
            "text": point.payload["text"],
            "title": point.payload.get("title", ""),
            "score": point.score,
        })

    return retrieved


def rerank(
    query: str,
    documents: list[dict],
    top_n: int = 3,
    api_key: str | None = None,
    model: str = "rerank-v3.5",
) -> list[dict]:
    """Rerank documents using Cohere.

    Falls back to returning top_n by original order if no API key.

    Args:
        query: The search query.
        documents: List of dicts with 'text' key.
        top_n: Number of documents to return.
        api_key: Cohere API key (falls back to env var).
        model: Cohere reranker model.

    Returns:
        Reranked list of document dicts.
    """
    cohere_key = api_key or os.getenv("COHERE_API_KEY", "")

    if cohere_key:
        try:
            import cohere
            co = cohere.ClientV2(api_key=cohere_key)
            doc_texts = [d["text"] for d in documents]
            response = co.rerank(
                model=model, query=query, documents=doc_texts, top_n=top_n,
            )
            reranked = []
            for r in response.results:
                doc = documents[r.index].copy()
                doc["rerank_score"] = r.relevance_score
                reranked.append(doc)
            return reranked
        except Exception:
            pass

    return documents[:top_n]
