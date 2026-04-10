"""Complete RAG pipeline: chunk -> embed -> index -> retrieve -> rerank -> generate."""

from __future__ import annotations

import time
from .config import RAGConfig
from .vectorstore import build_index
from .retriever import retrieve, rerank
from .generator import generate, DEFAULT_SYSTEM_PROMPT


class RAGPipeline:
    """Configurable RAG pipeline with Qdrant + OpenAI.

    Usage:
        from ragevals import RAGPipeline, RAGConfig
        from ragevals.datasets import ACME_KNOWLEDGE_BASE

        pipeline = RAGPipeline(documents=ACME_KNOWLEDGE_BASE)
        result = pipeline.run("What is your return policy?")
        print(result["answer"])
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        documents: list[dict] | None = None,
    ):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None).
            documents: Knowledge base documents with 'title' and 'content' keys.
                       Uses ACME_KNOWLEDGE_BASE if None.
        """
        from openai import OpenAI

        self.config = config or RAGConfig()
        self.openai_client = OpenAI()

        if documents is None:
            from .datasets import ACME_KNOWLEDGE_BASE
            documents = ACME_KNOWLEDGE_BASE

        self.documents = documents
        self.qdrant_client, self.collection_name, self.num_chunks = build_index(
            documents=documents,
            config=self.config,
            openai_client=self.openai_client,
        )

    def retrieve(self, query: str) -> list[str]:
        """Retrieve relevant context chunks for a query.

        Returns:
            List of context text strings.
        """
        results = retrieve(
            query=query,
            client=self.qdrant_client,
            collection_name=self.collection_name,
            top_k=self.config.top_k,
            embedding_model=self.config.embedding_model,
            openai_client=self.openai_client,
        )

        if self.config.use_reranker:
            results = rerank(
                query=query,
                documents=results,
                top_n=self.config.top_k,
                model=self.config.reranker_model,
            )

        return [r["text"] for r in results]

    def generate(self, query: str, contexts: list[str]) -> str:
        """Generate an answer from query and contexts."""
        return generate(
            query=query,
            contexts=contexts,
            client=self.openai_client,
            model=self.config.generation_model,
            temperature=self.config.temperature,
            system_prompt=self.config.system_prompt,
        )

    def run(self, query: str) -> dict:
        """Run the full RAG pipeline: retrieve then generate.

        Returns:
            Dict with keys: query, answer, contexts, latency_ms.
        """
        start = time.time()
        contexts = self.retrieve(query)
        answer = self.generate(query, contexts)
        latency_ms = (time.time() - start) * 1000
        return {
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "latency_ms": latency_ms,
        }

    def run_batch(self, queries: list[str]) -> list[dict]:
        """Run multiple queries through the pipeline."""
        return [self.run(q) for q in queries]
