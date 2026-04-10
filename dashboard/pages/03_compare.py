"""Compare Configs page.

Lets users configure two RAG pipelines side-by-side, evaluate the same
dataset with both, and see a comparison table and grouped bar chart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd

from ragevals.config import RAGConfig

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.header("Compare Configurations")
st.markdown("Evaluate two pipeline configurations on the same dataset and compare results.")

# ---------------------------------------------------------------------------
# Two-column config
# ---------------------------------------------------------------------------

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Config A")
    chunk_a = st.slider("Chunk Size (A)", 100, 2000, 500, key="cmp_chunk_a")
    topk_a = st.slider("Top-K (A)", 1, 20, 3, key="cmp_topk_a")
    temp_a = st.slider("Temperature (A)", 0.0, 1.0, 0.0, 0.1, key="cmp_temp_a")
    model_a = st.selectbox(
        "Model (A)", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], key="cmp_model_a",
    )
    rerank_a = st.checkbox("Use Reranker (A)", False, key="cmp_rerank_a")
    config_a = RAGConfig(
        chunk_size=chunk_a, top_k=topk_a, temperature=temp_a,
        generation_model=model_a, use_reranker=rerank_a,
    )

with col_right:
    st.subheader("Config B")
    chunk_b = st.slider("Chunk Size (B)", 100, 2000, 500, key="cmp_chunk_b")
    topk_b = st.slider("Top-K (B)", 1, 20, 5, key="cmp_topk_b")
    temp_b = st.slider("Temperature (B)", 0.0, 1.0, 0.0, 0.1, key="cmp_temp_b")
    model_b = st.selectbox(
        "Model (B)", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], key="cmp_model_b",
    )
    rerank_b = st.checkbox("Use Reranker (B)", True, key="cmp_rerank_b")
    config_b = RAGConfig(
        chunk_size=chunk_b, top_k=topk_b, temperature=temp_b,
        generation_model=model_b, use_reranker=rerank_b,
    )

# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Shared Dataset")

use_builtin = st.checkbox("Use built-in golden test cases", value=True, key="cmp_builtin")

data_dir = Path(__file__).resolve().parents[2] / "data"
dataset_files = sorted(data_dir.glob("golden_*.json")) if data_dir.exists() else []

selected_file = None
if not use_builtin and dataset_files:
    selected_file = st.selectbox(
        "Dataset file", [str(f) for f in dataset_files],
        format_func=lambda x: Path(x).name, key="cmp_sel",
    )

# ---------------------------------------------------------------------------
# Pipeline builder (cached)
# ---------------------------------------------------------------------------


@st.cache_resource
def _build_pipeline(_config_dict: dict):
    from ragevals.config import RAGConfig, load_env
    from ragevals.pipeline import RAGPipeline

    load_env()
    cfg = RAGConfig.from_dict(_config_dict)
    return RAGPipeline(cfg)


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

if st.button("Run Comparison", type="primary"):
    from ragevals.datasets import GOLDEN_TEST_CASES, load_dataset
    from ragevals.evaluation import evaluate_pipeline

    if use_builtin:
        test_cases = GOLDEN_TEST_CASES
    else:
        test_cases = load_dataset(selected_file)

    try:
        with st.spinner(f"Building & evaluating Config A ({config_a.name})..."):
            pa = _build_pipeline(vars(config_a))
            results_a = evaluate_pipeline(pa, test_cases)

        with st.spinner(f"Building & evaluating Config B ({config_b.name})..."):
            pb = _build_pipeline(vars(config_b))
            results_b = evaluate_pipeline(pb, test_cases)

        st.session_state["cmp_results_a"] = results_a
        st.session_state["cmp_results_b"] = results_b
        st.session_state["cmp_names"] = (config_a.name, config_b.name)
        st.success("Comparison complete!")

    except Exception as exc:
        st.error(f"Comparison failed: {exc}")

# ---------------------------------------------------------------------------
# Display comparison
# ---------------------------------------------------------------------------

if "cmp_results_a" in st.session_state:
    results_a = st.session_state["cmp_results_a"]
    results_b = st.session_state["cmp_results_b"]
    name_a, name_b = st.session_state["cmp_names"]

    s_a = results_a.get("summary", {})
    s_b = results_b.get("summary", {})
    all_metrics = sorted(set(s_a) | set(s_b))

    if all_metrics:
        rows = []
        for m in all_metrics:
            va = s_a.get(m, 0.0)
            vb = s_b.get(m, 0.0)
            delta = vb - va
            if delta > 0.01:
                winner = name_b
            elif delta < -0.01:
                winner = name_a
            else:
                winner = "Tie"
            rows.append({
                "Metric": m,
                name_a: round(va, 4),
                name_b: round(vb, 4),
                "Delta (B-A)": round(delta, 4),
                "Winner": winner,
            })

        st.subheader("Comparison Table")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Grouped bar chart.
        st.subheader("Visual Comparison")
        chart_df = pd.DataFrame({
            name_a: {m: s_a.get(m, 0) for m in all_metrics},
            name_b: {m: s_b.get(m, 0) for m in all_metrics},
        })
        st.bar_chart(chart_df)

    # Latency comparison.
    if "combined_df" in results_a and "latency_ms" in results_a["combined_df"]:
        st.subheader("Latency Comparison")
        lat_a = results_a["combined_df"]["latency_ms"].mean()
        lat_b = results_b["combined_df"]["latency_ms"].mean()
        lc1, lc2 = st.columns(2)
        lc1.metric(f"{name_a} Avg Latency", f"{lat_a:.0f} ms")
        lc2.metric(
            f"{name_b} Avg Latency", f"{lat_b:.0f} ms",
            delta=f"{lat_b - lat_a:+.0f} ms",
        )
