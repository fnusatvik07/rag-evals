"""Qdrant vector store management."""

from __future__ import annotations

from .embeddings import get_embeddings
from .chunking import chunk_documents


def create_collection(client, name: str, dim: int = 1536) -> None:
    """Create a Qdrant collection with cosine similarity."""
    from qdrant_client.models import Distance, VectorParams

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def upsert_chunks(
    client,
    collection_name: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> int:
    """Upsert chunks with embeddings into Qdrant.

    Returns:
        Number of points upserted.
    """
    from qdrant_client.models import PointStruct

    points = [
        PointStruct(
            id=i,
            vector=emb,
            payload={"text": chunk["text"], "title": chunk["title"],
                     "doc_index": chunk.get("doc_index", 0),
                     "chunk_index": chunk.get("chunk_index", 0)},
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def build_index(
    documents: list[dict],
    config=None,
    client=None,
    openai_client=None,
) -> tuple:
    """Build a Qdrant index from documents.

    High-level function: chunks documents, embeds, creates collection, upserts.

    Args:
        documents: List of dicts with 'title' and 'content'.
        config: RAGConfig (or uses defaults).
        client: Optional QdrantClient (creates in-memory if None).
        openai_client: Optional OpenAI client for embeddings.

    Returns:
        Tuple of (qdrant_client, collection_name, num_chunks).
    """
    if config is None:
        from .config import RAGConfig
        config = RAGConfig()

    if client is None:
        from qdrant_client import QdrantClient
        client = QdrantClient(":memory:")

    # Chunk documents
    chunks = chunk_documents(
        documents, chunk_size=config.chunk_size, overlap=config.chunk_overlap
    )

    # Embed
    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts, model=config.embedding_model, client=openai_client)

    # Create collection and upsert
    collection_name = config.collection_name
    create_collection(client, collection_name, dim=len(embeddings[0]))
    count = upsert_chunks(client, collection_name, chunks, embeddings)

    return client, collection_name, count
