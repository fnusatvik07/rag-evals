"""Sidebar configuration builder for Streamlit dashboards."""

import streamlit as st

# Ensure the ragevals package is importable (parent dir on sys.path).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ragevals.config import RAGConfig


def config_sidebar(key_prefix: str = "") -> RAGConfig:
    """Build a RAGConfig from sidebar widgets.

    Parameters
    ----------
    key_prefix : str
        A prefix for widget keys so that multiple sidebars can coexist
        (e.g., for the comparison page).

    Returns
    -------
    RAGConfig
        A populated configuration object.
    """
    st.sidebar.subheader("Pipeline Configuration")

    chunk_size = st.sidebar.slider(
        "Chunk Size", 100, 2000, 500, key=f"{key_prefix}chunk"
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap", 0, 200, 50, key=f"{key_prefix}overlap"
    )
    top_k = st.sidebar.slider(
        "Top-K", 1, 20, 3, key=f"{key_prefix}topk"
    )
    temperature = st.sidebar.slider(
        "Temperature", 0.0, 1.0, 0.0, 0.1, key=f"{key_prefix}temp"
    )
    model = st.sidebar.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        key=f"{key_prefix}model",
    )
    use_reranker = st.sidebar.checkbox(
        "Use Reranker", False, key=f"{key_prefix}rerank"
    )

    metrics_options = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "hallucination",
        "toxicity",
        "bias",
    ]
    selected_metrics = st.sidebar.multiselect(
        "Metrics",
        metrics_options,
        default=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        key=f"{key_prefix}metrics",
    )

    return RAGConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        temperature=temperature,
        generation_model=model,
        use_reranker=use_reranker,
    )
