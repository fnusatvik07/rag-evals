"""RAG Evaluation Dashboard -- main entry point.

Launch with::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the ragevals package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("RAG Evaluation Dashboard")
st.sidebar.markdown("---")

pages = {
    "Home": None,
    "Single Query": "dashboard/pages/01_query.py",
    "Batch Evaluation": "dashboard/pages/02_batch.py",
    "Compare Configs": "dashboard/pages/03_compare.py",
    "History": "dashboard/pages/04_history.py",
    "Datasets": "dashboard/pages/05_datasets.py",
}

page = st.sidebar.radio("Navigate", list(pages.keys()))

st.sidebar.markdown("---")
st.sidebar.caption("RAG Evals v0.1.0")

# ---------------------------------------------------------------------------
# Home / welcome page
# ---------------------------------------------------------------------------

if page == "Home":
    st.title("RAG Evaluation Dashboard")
    st.markdown(
        """
        Welcome to the **RAG Evaluation Dashboard**. Use the sidebar to navigate
        between pages.

        **Available pages:**

        | Page | Description |
        |------|-------------|
        | Single Query | Evaluate a single question through the RAG pipeline |
        | Batch Evaluation | Run evaluation on an entire dataset |
        | Compare Configs | Side-by-side comparison of two configurations |
        | History | Browse past evaluation runs, trends, and regressions |
        | Datasets | Manage evaluation datasets |
        """
    )

    # Overview stats from history (if available).
    st.markdown("---")
    st.subheader("Overview")

    try:
        from ragevals.history import EvaluationHistory

        hist = EvaluationHistory()
        runs = hist.get_runs(limit=100)
        hist.close()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Runs", len(runs))

        baseline_count = sum(1 for r in runs if r["is_baseline"])
        col2.metric("Baselines Set", baseline_count)

        if runs:
            latest = runs[0]
            avg_score = (
                sum(latest["scores"].values()) / len(latest["scores"])
                if latest["scores"]
                else 0.0
            )
            col3.metric("Latest Avg Score", f"{avg_score:.3f}")
        else:
            col3.metric("Latest Avg Score", "N/A")

        if runs:
            st.markdown("#### Recent Runs")
            import pandas as pd

            run_rows = []
            for r in runs[:10]:
                run_rows.append(
                    {
                        "ID": r["id"][:12],
                        "Timestamp": r["timestamp"][:19],
                        "Baseline": "Yes" if r["is_baseline"] else "",
                        **{k: f"{v:.3f}" for k, v in r["scores"].items()},
                    }
                )
            st.dataframe(pd.DataFrame(run_rows), use_container_width=True)
    except Exception as exc:
        st.info(f"No evaluation history available yet. Run some evaluations to populate this section. ({exc})")

# ---------------------------------------------------------------------------
# Sub-page routing
# ---------------------------------------------------------------------------

elif page == "Single Query":
    exec(open(str(Path(__file__).parent / "pages" / "01_query.py")).read())
elif page == "Batch Evaluation":
    exec(open(str(Path(__file__).parent / "pages" / "02_batch.py")).read())
elif page == "Compare Configs":
    exec(open(str(Path(__file__).parent / "pages" / "03_compare.py")).read())
elif page == "History":
    exec(open(str(Path(__file__).parent / "pages" / "04_history.py")).read())
elif page == "Datasets":
    exec(open(str(Path(__file__).parent / "pages" / "05_datasets.py")).read())
