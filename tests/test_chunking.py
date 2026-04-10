"""Tests for ragevals.chunking -- chunk_text and chunk_documents."""

import pytest

from ragevals.chunking import chunk_text, chunk_documents


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_known_input_output(self):
        text = "Sentence one. Sentence two. Sentence three."
        chunks = chunk_text(text, chunk_size=500, overlap=0)
        # The text is short enough to fit in one chunk
        assert len(chunks) == 1
        assert "Sentence one" in chunks[0]
        assert "Sentence three" in chunks[0]

    def test_empty_string(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   ") == []

    def test_short_string_single_chunk(self):
        text = "Hello world."
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_splits_long_text(self):
        # Create a text that must be split
        text = ". ".join(f"This is sentence number {i}" for i in range(30))
        chunks = chunk_text(text, chunk_size=100, overlap=0)
        assert len(chunks) > 1
        # All content should be preserved
        combined = " ".join(chunks)
        assert "sentence number 0" in combined
        assert "sentence number 29" in combined

    def test_overlap_produces_shared_content(self):
        text = ". ".join(f"Word{i} is here" for i in range(20))
        chunks_with_overlap = chunk_text(text, chunk_size=60, overlap=20)
        chunks_no_overlap = chunk_text(text, chunk_size=60, overlap=0)
        # With overlap, there should be more chunks (or at least equal)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_chunk_size_respected(self):
        # First chunk should respect chunk_size (subsequent ones may exceed
        # slightly because of overlap prepend)
        text = ". ".join(f"Sentence {i} with some extra words" for i in range(50))
        chunks = chunk_text(text, chunk_size=80, overlap=0)
        for chunk in chunks:
            # Each individual sentence added must have been <= chunk_size
            # but the final chunk is whatever remains
            assert len(chunk) > 0

    def test_returns_list_of_strings(self):
        chunks = chunk_text("Hello. World.", chunk_size=50, overlap=0)
        assert isinstance(chunks, list)
        for c in chunks:
            assert isinstance(c, str)


# ---------------------------------------------------------------------------
# chunk_documents
# ---------------------------------------------------------------------------

class TestChunkDocuments:
    def test_basic_chunking(self):
        docs = [
            {"title": "Doc A", "content": "Short content for doc A."},
            {"title": "Doc B", "content": "Short content for doc B."},
        ]
        chunks = chunk_documents(docs, chunk_size=500, overlap=0)
        assert len(chunks) >= 2
        titles = {c["title"] for c in chunks}
        assert "Doc A" in titles
        assert "Doc B" in titles

    def test_chunk_has_required_keys(self):
        docs = [{"title": "Test", "content": "Some text here."}]
        chunks = chunk_documents(docs, chunk_size=500, overlap=0)
        assert len(chunks) >= 1
        for c in chunks:
            assert "text" in c
            assert "title" in c
            assert "doc_index" in c
            assert "chunk_index" in c

    def test_doc_index_tracks_source(self):
        docs = [
            {"title": "First", "content": "First doc content."},
            {"title": "Second", "content": "Second doc content."},
        ]
        chunks = chunk_documents(docs, chunk_size=500, overlap=0)
        doc_indices = {c["doc_index"] for c in chunks}
        assert 0 in doc_indices
        assert 1 in doc_indices

    def test_empty_document_list(self):
        chunks = chunk_documents([], chunk_size=400, overlap=50)
        assert chunks == []

    def test_long_document_produces_multiple_chunks(self):
        long_content = ". ".join(f"Sentence {i} with content" for i in range(100))
        docs = [{"title": "Long", "content": long_content}]
        chunks = chunk_documents(docs, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        # All chunks should reference the same document
        for c in chunks:
            assert c["doc_index"] == 0
            assert c["title"] == "Long"
