"""
LangChain + RAGEvals Integration Example
=========================================

Builds a LangChain RAG chain with in-memory Qdrant and OpenAI, then evaluates
it using ragevals metrics. Fully self-contained -- just set OPENAI_API_KEY in
your .env file and run:

    python integrations/langchain_example.py

Requirements (beyond ragevals):
    pip install langchain langchain-openai langchain-community qdrant-client
"""

import os
import sys

# Ensure the ragevals package is importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import Qdrant
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
except ImportError:
    print(
        "ERROR: Missing LangChain dependencies.\n"
        "Install them with:\n"
        "  pip install langchain langchain-openai langchain-community qdrant-client\n"
    )
    sys.exit(1)

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Load environment
# ---------------------------------------------------------------------------
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your .env file"

# ---------------------------------------------------------------------------
# 2. Build Qdrant vectorstore with LangChain
# ---------------------------------------------------------------------------
DOCUMENTS = [
    Document(page_content=(
        "Acme Corp offers a 30-day return policy on all products. Items must be "
        "unused, in original packaging, with receipt. Refunds take 5-7 business days."
    ), metadata={"source": "returns"}),
    Document(page_content=(
        "Electronics have a 15-day return window. A 15% restocking fee may apply "
        "to opened items. Defective electronics can be exchanged within 90 days."
    ), metadata={"source": "electronics_returns"}),
    Document(page_content=(
        "Standard Shipping is free over $50 (5-7 days). Expedited is $12.99 (2-3 days). "
        "Overnight is $24.99 (next business day if ordered before 2 PM EST)."
    ), metadata={"source": "shipping"}),
    Document(page_content=(
        "The Acme SmartHome Hub costs $149.99, supports WiFi/Bluetooth/Zigbee/Z-Wave, "
        "has voice control, 5-inch touchscreen, and energy monitoring. 2-year warranty."
    ), metadata={"source": "products"}),
    Document(page_content=(
        "Acme Rewards: free to join, 1 point per dollar. 100 points = $5 off. "
        "Silver (500+ pts/yr) adds free expedited shipping. Gold (1000+) adds priority support."
    ), metadata={"source": "loyalty"}),
]

print("Building Qdrant vectorstore via LangChain...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Qdrant.from_documents(
    documents=DOCUMENTS,
    embedding=embeddings,
    location=":memory:",
    collection_name="langchain_demo",
)

# ---------------------------------------------------------------------------
# 3. Build RetrievalQA chain
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

print("RetrievalQA chain ready.\n")


# ---------------------------------------------------------------------------
# 4. Adapter class that wraps the chain for ragevals evaluation
# ---------------------------------------------------------------------------
class LangChainRAGAdapter:
    """Wraps a LangChain RetrievalQA chain to match the ragevals pipeline interface."""

    def __init__(self, chain):
        self.chain = chain

    def run(self, query: str) -> dict:
        result = self.chain.invoke({"query": query})
        contexts = [doc.page_content for doc in result.get("source_documents", [])]
        return {
            "query": query,
            "answer": result["result"],
            "contexts": contexts,
        }


adapter = LangChainRAGAdapter(qa_chain)

# ---------------------------------------------------------------------------
# 5. Define test cases and run evaluation
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "query": "What is the return policy for regular items?",
        "reference": "30-day return policy. Items must be unused, in original packaging, with receipt. Refunds in 5-7 business days.",
    },
    {
        "query": "How long do I have to return electronics?",
        "reference": "Electronics have a 15-day return window with a possible 15% restocking fee.",
    },
    {
        "query": "What shipping options are available?",
        "reference": "Standard (free over $50, 5-7 days), Expedited ($12.99, 2-3 days), Overnight ($24.99).",
    },
    {
        "query": "What are the features of the SmartHome Hub?",
        "reference": "Costs $149.99, supports WiFi/Bluetooth/Zigbee/Z-Wave, voice control, touchscreen, energy monitoring, 2-year warranty.",
    },
    {
        "query": "How does the loyalty program work?",
        "reference": "Free to join, 1 point per dollar, 100 pts = $5 off. Silver and Gold tiers with extra perks.",
    },
]

print("Running evaluation on 5 test cases...\n")
for i, tc in enumerate(TEST_CASES, 1):
    result = adapter.run(tc["query"])
    print(f"  [{i}] Q: {tc['query']}")
    print(f"      A: {result['answer'][:120]}...")
    print(f"      Contexts retrieved: {len(result['contexts'])}")
    print()

print("LangChain integration example complete.")
print("To run full metric evaluation, pass results to DeepEval or RAGAS (see workbooks/).")
