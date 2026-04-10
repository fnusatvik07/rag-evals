"""Built-in datasets and dataset I/O utilities."""

from __future__ import annotations

import glob as _glob
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Acme Corp knowledge base -- 15 documents
# ---------------------------------------------------------------------------

ACME_KNOWLEDGE_BASE: list[dict] = [
    {"id": 1, "title": "Return Policy Overview", "content": (
        "Acme Corp offers a 30-day return policy on all products purchased through "
        "our website or retail stores. To be eligible for a return, the item must be "
        "unused, in its original packaging, and accompanied by the original receipt or "
        "proof of purchase. Refunds are processed within 5-7 business days after we "
        "receive the returned item. Shipping costs for returns are the responsibility "
        "of the customer unless the item was defective or the wrong item was shipped. "
        "Items marked as 'Final Sale' cannot be returned or exchanged."
    )},
    {"id": 2, "title": "Electronics Return Policy", "content": (
        "Electronic products purchased from Acme Corp have a 15-day return window "
        "instead of the standard 30 days. All electronics must be returned with all "
        "original accessories, cables, manuals, and packaging. A restocking fee of 15% "
        "may apply to opened electronics. Defective electronics can be exchanged for "
        "the same model within the first 90 days of purchase at no additional cost. "
        "Software products and digital downloads are non-returnable once the seal is "
        "broken or the download code has been redeemed."
    )},
    {"id": 3, "title": "Shipping Policy", "content": (
        "Acme Corp offers several shipping options for domestic orders within the "
        "United States. Standard Shipping takes 5-7 business days and is free for "
        "orders over $50. Expedited Shipping takes 2-3 business days and costs $12.99. "
        "Overnight Shipping is available for $24.99 and delivers the next business day "
        "if ordered before 2 PM EST. Orders are processed within 1-2 business days."
    )},
    {"id": 4, "title": "International Shipping Policy", "content": (
        "Acme Corp ships internationally to over 50 countries. International Standard "
        "Shipping takes 10-21 business days and costs vary by destination. International "
        "Express Shipping takes 5-7 business days. All international orders may be "
        "subject to customs duties, taxes, and import fees which are the responsibility "
        "of the customer."
    )},
    {"id": 5, "title": "Product Warranty Information", "content": (
        "All Acme Corp branded products come with a 1-year limited warranty covering "
        "defects in materials and workmanship under normal use. The warranty does not "
        "cover damage caused by accidents, misuse, unauthorized modifications, or normal "
        "wear and tear. Extended warranty plans (2-year and 3-year) are available for "
        "purchase at the time of original purchase for an additional fee."
    )},
    {"id": 6, "title": "Customer Support Channels", "content": (
        "Acme Corp customer support is available through multiple channels. Phone support "
        "is available Monday through Friday, 8 AM to 8 PM EST at 1-800-ACME-HELP "
        "(1-800-226-3435). Email support can be reached at support@acmecorp.com with a "
        "typical response time of 24-48 hours. Live chat is available on our website "
        "Monday through Saturday, 9 AM to 6 PM EST."
    )},
    {"id": 7, "title": "Product: Acme SmartHome Hub", "content": (
        "The Acme SmartHome Hub is our flagship home automation controller priced at "
        "$149.99. It supports WiFi, Bluetooth, Zigbee, and Z-Wave protocols, making it "
        "compatible with over 10,000 smart home devices. Features include voice control, "
        "5-inch touchscreen display, energy monitoring, and automated routines. The "
        "SmartHome Hub comes with a 2-year warranty."
    )},
    {"id": 8, "title": "Product: Acme AirPure Pro Air Purifier", "content": (
        "The Acme AirPure Pro is a premium air purifier designed for rooms up to 800 "
        "square feet, priced at $299.99. It features a 4-stage filtration system: "
        "pre-filter, activated carbon filter, True HEPA H13 filter, and UV-C light "
        "sanitizer. The AirPure Pro removes 99.97% of particles as small as 0.3 microns."
    )},
    {"id": 9, "title": "Product: Acme FitBand Ultra", "content": (
        "The Acme FitBand Ultra is a fitness tracker priced at $79.99 that monitors "
        "heart rate, steps, sleep quality, blood oxygen levels, and stress. It features "
        "a 1.4-inch AMOLED display, 7-day battery life, and water resistance up to 50 "
        "meters."
    )},
    {"id": 10, "title": "Accepted Payment Methods", "content": (
        "Acme Corp accepts the following payment methods: Visa, MasterCard, American "
        "Express, Discover, PayPal, Apple Pay, Google Pay, and Acme Gift Cards. For "
        "orders over $200, we also offer Acme Pay Later, a buy-now-pay-later option."
    )},
    {"id": 11, "title": "Acme Rewards Loyalty Program", "content": (
        "The Acme Rewards program is free to join and earns members 1 point per dollar "
        "spent. Points can be redeemed for discounts: 100 points equals $5 off your next "
        "purchase. Silver tier (500+ points/year) unlocks free expedited shipping. Gold "
        "tier (1000+ points/year) adds priority customer support and exclusive product "
        "previews. Points expire after 12 months of account inactivity."
    )},
    {"id": 12, "title": "Privacy Policy Summary", "content": (
        "Acme Corp collects personal information including name, email, shipping address, "
        "payment information, and browsing behavior to process orders and improve our "
        "services. We do not sell personal information to third parties."
    )},
    {"id": 13, "title": "Product: Acme ErgoDesk Pro Standing Desk", "content": (
        "The Acme ErgoDesk Pro is a motorized standing desk priced at $599.99. It "
        "features a 60x30 inch bamboo desktop, dual-motor height adjustment from 25.5 "
        "to 51 inches, 4 programmable height presets, and anti-collision technology."
    )},
    {"id": 14, "title": "Order Cancellation Policy", "content": (
        "Orders can be cancelled within 1 hour of placement by contacting customer "
        "support or using the 'Cancel Order' button in your account dashboard. After "
        "1 hour, orders enter the processing queue and may not be cancellable."
    )},
    {"id": 15, "title": "Product: Acme CloudStorage Plans", "content": (
        "Acme CloudStorage offers secure cloud storage. Plans: Basic (50 GB) at "
        "$2.99/month, Standard (200 GB) at $5.99/month, Premium (2 TB) at $12.99/month. "
        "All plans include end-to-end encryption and automatic backup."
    )},
]

# ---------------------------------------------------------------------------
# Golden test cases -- 12 manually written Q&A pairs
# ---------------------------------------------------------------------------

GOLDEN_TEST_CASES: list[dict] = [
    {
        "query": "What is the return policy for regular items?",
        "reference": "Acme Corp offers a 30-day return policy. Items must be unused, in original packaging, with receipt. Refunds are processed in 5-7 business days.",
        "category": "returns",
    },
    {
        "query": "How long do I have to return electronics?",
        "reference": "Electronics have a 15-day return window. They must be returned with all original accessories and packaging. A 15% restocking fee may apply to opened items.",
        "category": "returns",
    },
    {
        "query": "What shipping options are available and how much do they cost?",
        "reference": "Acme Corp offers Standard Shipping (5-7 days, free over $50), Expedited Shipping (2-3 days, $12.99), and Overnight Shipping (next business day, $24.99).",
        "category": "shipping",
    },
    {
        "query": "Do you ship internationally?",
        "reference": "Yes, Acme Corp ships to over 50 countries. Standard international shipping takes 10-21 business days. Customers are responsible for customs duties and import fees.",
        "category": "shipping",
    },
    {
        "query": "What does the warranty cover on Acme products?",
        "reference": "Acme Corp products come with a 1-year limited warranty covering defects in materials and workmanship. Does not cover accidents, misuse, or normal wear.",
        "category": "warranty",
    },
    {
        "query": "How can I contact customer support?",
        "reference": "Phone (1-800-ACME-HELP, M-F 8AM-8PM EST), email (support@acmecorp.com, 24-48hr response), live chat (M-Sat 9AM-6PM EST).",
        "category": "support",
    },
    {
        "query": "What are the features of the Acme SmartHome Hub?",
        "reference": "The SmartHome Hub costs $149.99, supports WiFi/Bluetooth/Zigbee/Z-Wave, has voice control, 5-inch touchscreen, energy monitoring, and automated routines. 2-year warranty.",
        "category": "products",
    },
    {
        "query": "How much is the AirPure Pro and what does it filter?",
        "reference": "The AirPure Pro costs $299.99. 4-stage filtration (pre-filter, carbon, True HEPA H13, UV-C). Removes 99.97% of particles 0.3 microns or larger. Covers rooms up to 800 sq ft.",
        "category": "products",
    },
    {
        "query": "What payment methods do you accept?",
        "reference": "Visa, MasterCard, American Express, Discover, PayPal, Apple Pay, Google Pay, and Acme Gift Cards. Orders over $200 qualify for Acme Pay Later.",
        "category": "payments",
    },
    {
        "query": "How does the Acme Rewards program work?",
        "reference": "Free to join, 1 point per dollar. 100 points = $5 off. Silver tier (500+ pts) adds free expedited shipping. Gold tier (1000+ pts) adds priority support.",
        "category": "loyalty",
    },
    {
        "query": "What is the electronics restocking fee?",
        "reference": "A restocking fee of 15% may apply to opened electronics returned to Acme Corp.",
        "category": "returns",
    },
    {
        "query": "How long does it take to process a refund?",
        "reference": "Refunds are processed within 5-7 business days after Acme Corp receives the returned item.",
        "category": "returns",
    },
]


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path) -> list[dict]:
    """Load a JSON dataset from disk.

    Args:
        path: Path to a JSON file containing a list of dicts.

    Returns:
        The deserialized list of dicts.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_dataset(data: list[dict], path: str | Path) -> None:
    """Save a dataset to a JSON file.

    Args:
        data: List of dicts to serialize.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def validate_dataset(dataset: list[dict]) -> list[str]:
    """Validate that a test-case dataset has the required fields.

    Each item must have ``"query"`` and ``"reference"`` keys with non-empty
    string values.

    Args:
        dataset: List of test-case dicts.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []
    for idx, item in enumerate(dataset):
        if not isinstance(item, dict):
            errors.append(f"Item {idx}: not a dict")
            continue
        if "query" not in item or not item["query"].strip():
            errors.append(f"Item {idx}: missing or empty 'query'")
        if "reference" not in item or not item["reference"].strip():
            errors.append(f"Item {idx}: missing or empty 'reference'")
    return errors


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    docs_path: str,
    output_path: str,
    n_per_doc: int = 5,
    model: str = "gpt-4o-mini",
) -> str:
    """Generate a synthetic evaluation dataset from source documents.

    Reads documents from *docs_path*, uses an LLM to produce
    question / answer / context / ground_truth triples, and writes
    them to *output_path* as CSV.

    Args:
        docs_path: Directory containing source documents (TXT, MD, PDF).
        output_path: Destination CSV file path.
        n_per_doc: Number of test cases to generate per document.
        model: OpenAI model to use for generation.

    Returns:
        The *output_path* that was written.
    """
    import pandas as pd
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model,
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    doc_files: list[str] = []
    for ext in ("*.txt", "*.md", "*.pdf"):
        doc_files.extend(
            _glob.glob(os.path.join(docs_path, ext))
        )

    all_cases: list[dict] = []
    for doc_file in doc_files:
        with open(doc_file, "r", errors="ignore") as fh:
            content = fh.read()[:8000]

        prompt = (
            f"Given the following document, generate {n_per_doc} "
            "evaluation test cases.\n"
            "Each test case should have: query, answer, contexts "
            "(list of relevant passages), and ground_truth.\n\n"
            f"Document:\n{content}\n\n"
            "Return as a JSON array with keys: "
            "query, answer, contexts, ground_truth"
        )
        response = llm.invoke(prompt)
        try:
            cases = json.loads(response.content)
            if isinstance(cases, list):
                for case in cases:
                    case["source_doc"] = os.path.basename(doc_file)
                all_cases.extend(cases)
        except json.JSONDecodeError:
            continue

    df = pd.DataFrame(all_cases)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
