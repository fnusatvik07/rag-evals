"""
Haystack + RAGEvals Integration Example
========================================

Builds a Haystack RAG pipeline with an InMemoryDocumentStore,
OpenAI embedder, and generator, then evaluates it using ragevals.

    python integrations/haystack_example.py

Requirements (beyond ragevals):
    pip install haystack-ai
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from haystack import Document, Pipeline
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.embedders import (
        OpenAITextEmbedder,
        OpenAIDocumentEmbedder,
    )
    from haystack.components.retrievers.in_memory import (
        InMemoryEmbeddingRetriever,
    )
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator
except ImportError:
    print(
        "ERROR: Missing Haystack dependencies.\n"
        "Install them with:\n"
        "  pip install haystack-ai\n"
    )
    sys.exit(1)

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Load environment
# ---------------------------------------------------------------------------
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your .env file"

# ---------------------------------------------------------------------------
# 2. Build document store and embed documents
# ---------------------------------------------------------------------------
DOCUMENTS = [
    Document(content=(
        "Acme Corp offers a 30-day return policy on all products. Items must be "
        "unused, in original packaging, with receipt. Refunds take 5-7 business days."
    ), meta={"source": "returns"}),
    Document(content=(
        "Electronics have a 15-day return window. A 15% restocking fee may apply "
        "to opened items. Defective electronics can be exchanged within 90 days."
    ), meta={"source": "electronics_returns"}),
    Document(content=(
        "Standard Shipping is free over $50 (5-7 days). Expedited is $12.99 (2-3 days). "
        "Overnight is $24.99 (next business day if ordered before 2 PM EST)."
    ), meta={"source": "shipping"}),
    Document(content=(
        "The Acme SmartHome Hub costs $149.99, supports WiFi/Bluetooth/Zigbee/Z-Wave, "
        "has voice control, 5-inch touchscreen, and energy monitoring. 2-year warranty."
    ), meta={"source": "products"}),
    Document(content=(
        "Acme Rewards: free to join, 1 point per dollar. 100 points = $5 off. "
        "Silver (500+ pts/yr) adds free expedited shipping. Gold (1000+) adds priority support."
    ), meta={"source": "loyalty"}),
]

print("Building Haystack document store and embedding documents...")
document_store = InMemoryDocumentStore()

doc_embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
result = doc_embedder.run(documents=DOCUMENTS)
document_store.write_documents(result["documents"])
print(f"Indexed {document_store.count_documents()} documents.\n")

# ---------------------------------------------------------------------------
# 3. Build the Haystack RAG pipeline
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """
Answer the question based on the provided context. If the answer is not in
the context, say so. Be concise and accurate.

Context:
{% for doc in documents %}
  - {{ doc.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", OpenAITextEmbedder(model="text-embedding-3-small"))
rag_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=3))
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE))
rag_pipeline.add_component("generator", OpenAIGenerator(model="gpt-4o-mini"))

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "generator")

print("Haystack RAG pipeline built.\n")


# ---------------------------------------------------------------------------
# 4. Adapter class for ragevals evaluation interface
# ---------------------------------------------------------------------------
class HaystackRAGAdapter:
    """Wraps a Haystack Pipeline to match the ragevals pipeline interface."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self, query: str) -> dict:
        result = self.pipeline.run(
            {
                "text_embedder": {"text": query},
                "prompt_builder": {"question": query},
            }
        )
        answer = result["generator"]["replies"][0]
        contexts = [
            doc.content
            for doc in result.get("retriever", {}).get("documents", [])
        ]
        return {
            "query": query,
            "answer": answer,
            "contexts": contexts,
        }


adapter = HaystackRAGAdapter(rag_pipeline)

# ---------------------------------------------------------------------------
# 5. Define test cases and run evaluation
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "query": "What is the return policy for regular items?",
        "reference": "30-day return policy. Unused, original packaging, with receipt. Refunds in 5-7 business days.",
    },
    {
        "query": "How long do I have to return electronics?",
        "reference": "15-day return window. 15% restocking fee may apply to opened items.",
    },
    {
        "query": "What shipping options are available?",
        "reference": "Standard (free over $50), Expedited ($12.99), Overnight ($24.99).",
    },
    {
        "query": "What are the features of the SmartHome Hub?",
        "reference": "$149.99, WiFi/Bluetooth/Zigbee/Z-Wave, voice control, touchscreen, 2-year warranty.",
    },
    {
        "query": "How does the loyalty program work?",
        "reference": "1 point per dollar, 100 pts = $5 off. Silver and Gold tiers.",
    },
]

print("Running evaluation on 5 test cases...\n")
for i, tc in enumerate(TEST_CASES, 1):
    result = adapter.run(tc["query"])
    print(f"  [{i}] Q: {tc['query']}")
    print(f"      A: {result['answer'][:120]}...")
    print(f"      Contexts retrieved: {len(result['contexts'])}")
    print()

print("Haystack integration example complete.")
print("To run full metric evaluation, pass results to DeepEval or RAGAS (see workbooks/).")
