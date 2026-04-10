"""Single Query Evaluation page.

Lets users enter a question, run the RAG pipeline, and view the answer
alongside per-metric score cards.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import numpy as np

from ragevals.config import RAGConfig
from dashboard.components.sidebar import config_sidebar
from dashboard.components.metric_card import metric_card

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.header("Single Query Evaluation")
st.markdown("Evaluate a single question through the RAG pipeline and inspect metric scores.")

# ---------------------------------------------------------------------------
# Sidebar config
# ---------------------------------------------------------------------------

config = config_sidebar(key_prefix="sq_")

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

query = st.text_area("Enter your question:", height=80, placeholder="What is the return policy?")

with st.expander("Optional: provide ground truth"):
    ground_truth = st.text_input("Ground truth answer (optional):")

# ---------------------------------------------------------------------------
# Pipeline builder (cached)
# ---------------------------------------------------------------------------


@st.cache_resource
def _build_pipeline(_config_dict: dict):
    """Build a RAGPipeline from a config dict."""
    from ragevals.config import RAGConfig, load_env
    from ragevals.pipeline import RAGPipeline

    load_env()
    cfg = RAGConfig.from_dict(_config_dict)
    return RAGPipeline(cfg)


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

if st.button("Evaluate", type="primary"):
    if not query.strip():
        st.error("Please enter a question.")
    else:
        try:
            with st.spinner("Running pipeline..."):
                pipeline = _build_pipeline(vars(config))
                result = pipeline.run(query)

            # Display answer.
            st.subheader("Generated Answer")
            st.markdown(result["answer"])

            # Display retrieved contexts.
            if result["contexts"]:
                with st.expander(f"Retrieved Contexts ({len(result['contexts'])})"):
                    for i, ctx in enumerate(result["contexts"]):
                        st.markdown(f"**Context {i + 1}:**")
                        st.text(ctx)
                        st.markdown("---")

            st.info(f"Latency: {result['latency_ms']:.0f} ms")

            # Run evaluation metrics.
            with st.spinner("Running evaluation metrics..."):
                eval_data = [{
                    "query": query,
                    "response": result["answer"],
                    "reference": ground_truth if ground_truth and ground_truth.strip() else "",
                    "contexts": result["contexts"],
                }]

                from ragevals.evaluation import run_deepeval
                try:
                    scores_df = run_deepeval(eval_data)
                    metric_cols = [c for c in scores_df.columns if c != "query"]

                    st.subheader("Metric Scores")
                    cols = st.columns(min(len(metric_cols), 4))
                    for idx, col_name in enumerate(metric_cols):
                        with cols[idx % len(cols)]:
                            score = float(scores_df[col_name].iloc[0])
                            if not np.isnan(score):
                                metric_card(name=col_name, score=score)

                    st.session_state["last_single_result"] = {
                        "query": query,
                        "answer": result["answer"],
                        "scores": {c: float(scores_df[c].iloc[0]) for c in metric_cols},
                    }
                except Exception as eval_exc:
                    st.warning(f"Metric evaluation failed: {eval_exc}")
                    st.info("Pipeline ran successfully but metric scoring requires DeepEval to be configured.")

        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
