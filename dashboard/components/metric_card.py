"""Reusable metric display card for Streamlit dashboards."""

import streamlit as st


def metric_card(name: str, score: float, threshold: float = 0.7, reason: str = None):
    """Display a metric as a colored card.

    Parameters
    ----------
    name : str
        The metric name to display.
    score : float
        The metric score (0.0 - 1.0).
    threshold : float
        Score threshold for pass/fail coloring.
    reason : str, optional
        Explanation or reasoning behind the score.
    """
    passed = score >= threshold
    color = "#4CAF50" if passed else "#F44336"
    st.markdown(f"""
    <div style="border:2px solid {color}; border-radius:10px; padding:15px; margin:5px 0;">
        <h4 style="margin:0;">{name}</h4>
        <h2 style="color:{color}; margin:5px 0;">{score:.3f}</h2>
        <p style="margin:0;">{"PASS" if passed else "FAIL"} (threshold: {threshold})</p>
    </div>
    """, unsafe_allow_html=True)
    if reason:
        with st.expander("View reasoning"):
            st.write(reason)
