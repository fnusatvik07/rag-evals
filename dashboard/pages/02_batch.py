"""Batch Evaluation page.

Upload or select a dataset, run evaluation across all test cases,
and view results as a table, summary statistics, and bar chart.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd

from ragevals.config import RAGConfig
from dashboard.components.sidebar import config_sidebar

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.header("Batch Evaluation")
st.markdown("Evaluate an entire dataset and review aggregate results.")

# ---------------------------------------------------------------------------
# Sidebar config
# ---------------------------------------------------------------------------

config = config_sidebar(key_prefix="batch_")

# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------

st.subheader("Select Dataset")

data_dir = Path(__file__).resolve().parents[2] / "data"
existing_files = sorted(data_dir.glob("*.json")) if data_dir.exists() else []

tab_upload, tab_existing, tab_builtin = st.tabs(["Upload Dataset", "Use Existing", "Built-in"])

dataset_source = None
test_cases = None

with tab_upload:
    uploaded = st.file_uploader("Upload JSON", type=["json"], key="batch_upload")
    if uploaded is not None:
        test_cases = json.loads(uploaded.read())
        dataset_source = uploaded.name
        st.success(f"Uploaded: {uploaded.name} ({len(test_cases)} cases)")

with tab_existing:
    if existing_files:
        selected = st.selectbox(
            "Choose a dataset",
            [str(f) for f in existing_files],
            format_func=lambda x: Path(x).name,
            key="batch_existing",
        )
        if selected and test_cases is None:
            with open(selected) as fh:
                test_cases = json.load(fh)
            dataset_source = Path(selected).name
    else:
        st.info("No datasets found in data/ directory.")

with tab_builtin:
    if st.button("Use Golden Test Cases"):
        from ragevals.datasets import GOLDEN_TEST_CASES
        test_cases = GOLDEN_TEST_CASES
        dataset_source = "golden_test_cases (built-in)"
        st.success(f"Loaded {len(test_cases)} built-in test cases")

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
# Run evaluation
# ---------------------------------------------------------------------------

if st.button("Run Batch Evaluation", type="primary"):
    if test_cases is None:
        st.error("Please select or upload a dataset first.")
    else:
        try:
            with st.spinner("Building pipeline..."):
                pipeline = _build_pipeline(vars(config))

            with st.spinner(f"Evaluating {len(test_cases)} test cases... this may take a while."):
                from ragevals.evaluation import evaluate_pipeline
                results = evaluate_pipeline(pipeline, test_cases)

            if "combined_df" in results:
                st.session_state["batch_combined_df"] = results["combined_df"]
                st.session_state["batch_summary"] = results.get("summary", {})
            st.success("Evaluation complete!")

        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

if "batch_combined_df" in st.session_state:
    combined = st.session_state["batch_combined_df"]
    summary = st.session_state.get("batch_summary", {})

    metric_cols = [c for c in combined.columns if c.startswith("de_") or c.startswith("ragas_")]

    if metric_cols:
        st.subheader("Summary Statistics")
        summary_rows = []
        for col in metric_cols:
            vals = combined[col].dropna()
            status = "PASS" if vals.mean() >= 0.7 else "FAIL"
            summary_rows.append({
                "Metric": col,
                "Mean": f"{vals.mean():.4f}",
                "Min": f"{vals.min():.4f}",
                "Max": f"{vals.max():.4f}",
                "Std": f"{vals.std():.4f}",
                "Status": status,
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        # Bar chart of mean scores.
        st.subheader("Mean Scores by Metric")
        chart_data = pd.DataFrame({
            "Metric": metric_cols,
            "Mean": [combined[c].mean() for c in metric_cols],
        }).set_index("Metric")
        st.bar_chart(chart_data)

    # Full results table.
    st.subheader("Per-Query Results")
    st.dataframe(combined, use_container_width=True)

    # CSV download.
    csv_data = combined.to_csv(index=False)
    st.download_button(
        "Download Results CSV",
        data=csv_data,
        file_name="batch_eval_results.csv",
        mime="text/csv",
    )

    # Save to history.
    if st.button("Save to History") and metric_cols:
        try:
            from ragevals.history import EvaluationHistory

            # Reshape to long format for history.
            long_rows = []
            for idx, row in combined.iterrows():
                for metric in metric_cols:
                    long_rows.append({
                        "test_index": idx,
                        "query": row.get("query", f"Query {idx}"),
                        "metric_name": metric,
                        "score": float(row[metric]),
                    })
            scores_long = pd.DataFrame(long_rows)

            hist = EvaluationHistory()
            run_id = hist.save_run(
                config=vars(config),
                scores_df=scores_long,
                metadata={"source": "dashboard_batch", "dataset": dataset_source or "unknown"},
            )
            hist.close()
            st.success(f"Saved as run {run_id[:12]}...")
        except Exception as exc:
            st.error(f"Failed to save: {exc}")
