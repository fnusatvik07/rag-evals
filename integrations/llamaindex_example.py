"""
LlamaIndex + RAGEvals Integration Example
==========================================

Builds a LlamaIndex RAG pipeline with VectorStoreIndex, wraps the query
engine for evaluation, and runs ragevals evaluation on test cases.

    python integrations/llamaindex_example.py

Requirements (beyond ragevals):
    pip install llama-index llama-index-llms-openai llama-index-embeddings-openai
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError:
    print(
        "ERROR: Missing LlamaIndex dependencies.\n"
        "Install them with:\n"
        "  pip install llama-index llama-index-llms-openai "
        "llama-index-embeddings-openai\n"
    )
    sys.exit(1)

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Load environment
# ---------------------------------------------------------------------------
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your .env file"

# ---------------------------------------------------------------------------
# 2. Configure LlamaIndex settings
# ---------------------------------------------------------------------------
Settings.llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0.0)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

# ---------------------------------------------------------------------------
# 3. Build VectorStoreIndex from documents
# ---------------------------------------------------------------------------
DOCUMENTS = [
    Document(text=(
        "Acme Corp offers a 30-day return policy on all products. Items must be "
        "unused, in original packaging, with receipt. Refunds take 5-7 business days."
    ), metadata={"source": "returns"}),
    Document(text=(
        "Electronics have a 15-day return window. A 15% restocking fee may apply "
        "to opened items. Defective electronics can be exchanged within 90 days."
    ), metadata={"source": "electronics_returns"}),
    Document(text=(
        "Standard Shipping is free over $50 (5-7 days). Expedited is $12.99 (2-3 days). "
        "Overnight is $24.99 (next business day if ordered before 2 PM EST)."
    ), metadata={"source": "shipping"}),
    Document(text=(
        "The Acme SmartHome Hub costs $149.99, supports WiFi/Bluetooth/Zigbee/Z-Wave, "
        "has voice control, 5-inch touchscreen, and energy monitoring. 2-year warranty."
    ), metadata={"source": "products"}),
    Document(text=(
        "Acme Rewards: free to join, 1 point per dollar. 100 points = $5 off. "
        "Silver (500+ pts/yr) adds free expedited shipping. Gold (1000+) adds priority support."
    ), metadata={"source": "loyalty"}),
]

print("Building LlamaIndex VectorStoreIndex...")
index = VectorStoreIndex.from_documents(DOCUMENTS)
query_engine = index.as_query_engine(similarity_top_k=3)
print("Index built and query engine ready.\n")


# ---------------------------------------------------------------------------
# 4. Adapter class for ragevals evaluation interface
# ---------------------------------------------------------------------------
class LlamaIndexRAGAdapter:
    """Wraps a LlamaIndex query engine to match the ragevals pipeline interface."""

    def __init__(self, engine):
        self.engine = engine

    def run(self, query: str) -> dict:
        response = self.engine.query(query)
        contexts = []
        if hasattr(response, "source_nodes"):
            contexts = [node.get_content() for node in response.source_nodes]
        return {
            "query": query,
            "answer": str(response),
            "contexts": contexts,
        }


adapter = LlamaIndexRAGAdapter(query_engine)

# ---------------------------------------------------------------------------
# 5. Define test cases and run evaluation
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "query": "What is the return policy for regular items?",
        "reference": "30-day return policy. Items must be unused, in original packaging, with receipt.",
    },
    {
        "query": "How long do I have to return electronics?",
        "reference": "Electronics have a 15-day return window with a possible 15% restocking fee.",
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
    print(f"      A: {str(result['answer'])[:120]}...")
    print(f"      Contexts retrieved: {len(result['contexts'])}")
    print()

print("LlamaIndex integration example complete.")
print("To run full metric evaluation, pass results to DeepEval or RAGAS (see workbooks/).")
