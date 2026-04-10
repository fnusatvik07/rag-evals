"""History page.

Browse past evaluation runs, view trend charts, detect regressions,
and drill down into individual run details.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd

from ragevals.history import EvaluationHistory

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.header("Evaluation History")
st.markdown("Browse past runs, track metric trends, and detect regressions against your baseline.")

# ---------------------------------------------------------------------------
# Load history
# ---------------------------------------------------------------------------


@st.cache_resource(ttl=30)
def _get_history():
    """Return an EvaluationHistory instance (cached briefly)."""
    return EvaluationHistory()


hist = _get_history()

# ---------------------------------------------------------------------------
# Timeline of runs
# ---------------------------------------------------------------------------

st.subheader("Recent Runs")
limit = st.slider("Number of runs to show", 5, 100, 20, key="hist_limit")

try:
    runs = hist.get_runs(limit=limit)
except Exception as exc:
    st.error(f"Failed to load history: {exc}")
    runs = []

if not runs:
    st.info("No evaluation runs found. Run some evaluations first.")
else:
    run_rows = []
    for r in runs:
        score_str = ", ".join(f"{k}={v:.3f}" for k, v in r["scores"].items())
        run_rows.append({
            "ID": r["id"][:12],
            "Full ID": r["id"],
            "Timestamp": r["timestamp"][:19],
            "Baseline": "Yes" if r["is_baseline"] else "",
            "Scores": score_str,
        })

    runs_df = pd.DataFrame(run_rows)
    st.dataframe(runs_df[["ID", "Timestamp", "Baseline", "Scores"]], use_container_width=True)

    # ------------------------------------------------------------------
    # Trend chart
    # ------------------------------------------------------------------

    st.markdown("---")
    st.subheader("Metric Trends")

    # Collect all metric names across all runs.
    all_metrics = set()
    for r in runs:
        all_metrics.update(r["scores"].keys())
    all_metrics = sorted(all_metrics)

    if all_metrics:
        selected_metric = st.selectbox("Select metric for trend", all_metrics, key="hist_trend_metric")
        trend_data = hist.get_trend(selected_metric, last_n=limit)

        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"])
            trend_df = trend_df.set_index("timestamp")
            st.line_chart(trend_df[["mean_score"]])
        else:
            st.info(f"No trend data for metric '{selected_metric}'.")

    # ------------------------------------------------------------------
    # Regression detection
    # ------------------------------------------------------------------

    st.markdown("---")
    st.subheader("Regression Detection")

    if runs:
        latest = runs[0]
        if latest["scores"]:
            regression = hist.detect_regression(latest["scores"])

            if regression["baseline_id"] is None:
                st.info("No baseline set. Use the controls below to set one.")
            else:
                st.markdown(f"**Comparing latest run against baseline:** `{regression['baseline_id'][:12]}...`")

                if regression["overall_passed"]:
                    st.success("All metrics within threshold -- no regressions detected.")
                else:
                    st.error("Regressions detected!")

                if regression["regressions"]:
                    st.markdown("**Regressions:**")
                    for item in regression["regressions"]:
                        st.markdown(
                            f"- **{item['metric']}**: {item['current']:.4f} "
                            f"(baseline: {item['baseline']:.4f}, delta: {item['delta']:+.4f})"
                        )

                if regression["improvements"]:
                    st.markdown("**Improvements:**")
                    for item in regression["improvements"]:
                        st.markdown(
                            f"- **{item['metric']}**: {item['current']:.4f} "
                            f"(baseline: {item['baseline']:.4f}, delta: {item['delta']:+.4f})"
                        )

    # ------------------------------------------------------------------
    # Set baseline
    # ------------------------------------------------------------------

    st.markdown("---")
    st.subheader("Set Baseline")

    if runs:
        baseline_id = st.selectbox(
            "Select run to set as baseline",
            [r["id"] for r in runs],
            format_func=lambda x: f"{x[:12]}... ({next((r['timestamp'][:19] for r in runs if r['id'] == x), '')})",
            key="hist_baseline_select",
        )
        if st.button("Set as Baseline"):
            try:
                hist.set_baseline(baseline_id)
                st.success(f"Baseline set to {baseline_id[:12]}...")
                st.cache_resource.clear()
            except Exception as exc:
                st.error(f"Failed to set baseline: {exc}")

    # ------------------------------------------------------------------
    # Run detail drill-down
    # ------------------------------------------------------------------

    st.markdown("---")
    st.subheader("Run Detail")

    if runs:
        detail_id = st.selectbox(
            "Select run for details",
            [r["id"] for r in runs],
            format_func=lambda x: f"{x[:12]}... ({next((r['timestamp'][:19] for r in runs if r['id'] == x), '')})",
            key="hist_detail_select",
        )
        if st.button("Load Details"):
            try:
                detail = hist.get_run_detail(detail_id)

                st.markdown(f"**Run ID:** `{detail['id']}`")
                st.markdown(f"**Timestamp:** {detail['timestamp']}")
                st.markdown(f"**Baseline:** {'Yes' if detail['is_baseline'] else 'No'}")

                st.markdown("**Config:**")
                st.json(detail["config"])

                if detail["summary_scores"]:
                    st.markdown("**Summary Scores:**")
                    st.dataframe(pd.DataFrame(detail["summary_scores"]), use_container_width=True)

                if detail["details"]:
                    st.markdown("**Per-Query Details:**")
                    details_df = pd.DataFrame(detail["details"])
                    st.dataframe(details_df, use_container_width=True)

            except Exception as exc:
                st.error(f"Failed to load details: {exc}")
