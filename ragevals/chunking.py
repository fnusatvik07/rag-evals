"""Text chunking utilities for RAG pipelines."""

from __future__ import annotations

import hashlib


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into chunks with sentence-boundary awareness.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Characters of overlap from end of previous chunk.

    Returns:
        List of text chunks.
    """
    sentences = text.replace(". ", ".\n").split("\n")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[dict]:
    """Chunk a list of documents, preserving metadata.

    Args:
        documents: List of dicts with 'title' and 'content' keys.
        chunk_size: Maximum characters per chunk.
        overlap: Characters of overlap.

    Returns:
        List of chunk dicts with keys: text, title, doc_index, chunk_index, id.
    """
    all_chunks = []
    for doc_index, doc in enumerate(documents):
        title = doc.get("title", f"Document {doc_index}")
        content = doc.get("content", "")
        full_text = f"{title}\n\n{content}" if len(content) > chunk_size else content

        chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{title}_{chunk_index}".encode()).hexdigest()
            all_chunks.append({
                "id": chunk_id,
                "text": chunk,
                "title": title,
                "doc_index": doc_index,
                "chunk_index": chunk_index,
            })

    return all_chunks
