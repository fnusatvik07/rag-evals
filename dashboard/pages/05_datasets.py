"""Dataset Management page.

Browse datasets in the ``data/`` directory, preview test cases,
upload new datasets, and download existing ones.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.header("Dataset Management")
st.markdown("Browse, upload, and download evaluation datasets from the `data/` directory.")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Ensure the data directory exists.
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Browse existing datasets
# ---------------------------------------------------------------------------

st.subheader("Existing Datasets")

csv_files = sorted(DATA_DIR.glob("*.csv"))
json_files = sorted(DATA_DIR.glob("*.json"))
all_files = csv_files + json_files

if not all_files:
    st.info("No datasets found in data/. Upload one below.")
else:
    file_info = []
    for fp in all_files:
        stat = fp.stat()
        file_info.append({
            "Name": fp.name,
            "Type": fp.suffix,
            "Size (KB)": round(stat.st_size / 1024, 1),
            "Path": str(fp),
        })

    info_df = pd.DataFrame(file_info)
    st.dataframe(info_df[["Name", "Type", "Size (KB)"]], use_container_width=True)

    # ------------------------------------------------------------------
    # Preview selected dataset
    # ------------------------------------------------------------------

    st.markdown("---")
    st.subheader("Preview Dataset")

    selected_name = st.selectbox(
        "Select a dataset to preview",
        [fi["Name"] for fi in file_info],
        key="ds_preview_select",
    )

    if selected_name:
        selected_path = DATA_DIR / selected_name
        try:
            if selected_path.suffix == ".json":
                df = pd.read_json(selected_path)
            else:
                df = pd.read_csv(selected_path)

            st.markdown(f"**Rows:** {len(df)} | **Columns:** {', '.join(df.columns)}")
            st.dataframe(df.head(50), use_container_width=True)

            # Show individual test case detail.
            if "query" in df.columns and len(df) > 0:
                st.markdown("#### Test Case Detail")
                case_idx = st.number_input(
                    "Select row index",
                    min_value=0,
                    max_value=len(df) - 1,
                    value=0,
                    key="ds_case_idx",
                )
                case = df.iloc[case_idx]
                for col in df.columns:
                    st.markdown(f"**{col}:** {case[col]}")

        except Exception as exc:
            st.error(f"Failed to read {selected_name}: {exc}")

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    st.markdown("---")
    st.subheader("Download Dataset")

    dl_name = st.selectbox(
        "Select a dataset to download",
        [fi["Name"] for fi in file_info],
        key="ds_download_select",
    )

    if dl_name:
        dl_path = DATA_DIR / dl_name
        with open(dl_path, "rb") as fh:
            dl_data = fh.read()
        mime = "text/csv" if dl_path.suffix == ".csv" else "application/json"
        st.download_button(
            f"Download {dl_name}",
            data=dl_data,
            file_name=dl_name,
            mime=mime,
        )

# ---------------------------------------------------------------------------
# Upload new dataset
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Upload New Dataset")

uploaded = st.file_uploader(
    "Upload a CSV or JSON file",
    type=["csv", "json"],
    key="ds_upload",
)

if uploaded is not None:
    dest = DATA_DIR / uploaded.name
    if dest.exists():
        st.warning(f"A file named '{uploaded.name}' already exists. It will be overwritten.")

    if st.button("Save Uploaded File"):
        try:
            dest.write_bytes(uploaded.read())
            st.success(f"Saved to {dest}")
            st.cache_resource.clear()
        except Exception as exc:
            st.error(f"Failed to save: {exc}")

# ---------------------------------------------------------------------------
# Generate synthetic dataset
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Built-in Datasets")

with st.expander("View built-in knowledge base and golden test cases"):
    from ragevals.datasets import ACME_KNOWLEDGE_BASE, GOLDEN_TEST_CASES

    st.markdown(f"**Knowledge Base:** {len(ACME_KNOWLEDGE_BASE)} documents")
    kb_rows = [
        {"ID": d["id"], "Title": d["title"], "Preview": d["content"][:80] + "..."}
        for d in ACME_KNOWLEDGE_BASE
    ]
    st.dataframe(pd.DataFrame(kb_rows), use_container_width=True)

    st.markdown(f"**Golden Test Cases:** {len(GOLDEN_TEST_CASES)} cases")
    tc_rows = [
        {"Query": tc["query"][:60], "Category": tc.get("category", "")}
        for tc in GOLDEN_TEST_CASES
    ]
    st.dataframe(pd.DataFrame(tc_rows), use_container_width=True)
