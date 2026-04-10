"""Retrieval and reranking functions."""

from __future__ import annotations

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
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[dict]:
    """Rerank documents using a cross-encoder model from sentence-transformers.

    Runs locally — no API key required.

    Args:
        query: The search query.
        documents: List of dicts with 'text' key.
        top_n: Number of documents to return.
        model: Cross-encoder model name from HuggingFace.

    Returns:
        Reranked list of document dicts (highest relevance first).
    """
    if not documents:
        return []

    try:
        from sentence_transformers import CrossEncoder

        cross_encoder = CrossEncoder(model)
        pairs = [[query, d["text"]] for d in documents]
        scores = cross_encoder.predict(pairs)

        scored_docs = []
        for idx, score in enumerate(scores):
            doc = documents[idx].copy()
            doc["rerank_score"] = float(score)
            scored_docs.append(doc)

        scored_docs.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored_docs[:top_n]

    except ImportError:
        # Fallback if sentence-transformers not available.
        return documents[:top_n]
