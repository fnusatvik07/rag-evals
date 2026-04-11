#!/usr/bin/env python3
"""
Restructure 11 notebooks for classroom teaching.

Reads all source notebooks from workbooks/, extracts/transforms cells,
writes 11 restructured notebooks to workbooks_v2/.

Target: ~25-35 cells per notebook, one topic per class session,
side-by-side DeepEval+RAGAS comparisons, no duplicate pipeline rebuilds.
"""

import json
import copy
import os
import shutil

SRC = "workbooks_backup"
DST = "workbooks"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_nb(name):
    """Load a notebook by filename (without directory)."""
    path = os.path.join(SRC, name)
    with open(path) as f:
        return json.load(f)


def save_nb(nb, name):
    """Save a notebook to DST directory."""
    path = os.path.join(DST, name)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Saved {name} — {len(nb['cells'])} cells")


def empty_nb(metadata=None):
    """Create an empty notebook skeleton."""
    return {
        "cells": [],
        "metadata": metadata or {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def copy_cell(cell):
    """Deep-copy a cell, clearing outputs and execution_count."""
    c = copy.deepcopy(cell)
    if c.get("cell_type") == "code":
        c["outputs"] = []
        c["execution_count"] = None
    # Remove cell id if present (not required)
    c.pop("id", None)
    return c


def copy_cells(nb, indices):
    """Copy a list of cells by index from a notebook."""
    return [copy_cell(nb["cells"][i]) for i in indices]


def new_md(source):
    """Create a new markdown cell."""
    if isinstance(source, list):
        lines = source
    else:
        lines = source.split("\n")
    # Ensure each line ends with \n except the last
    formatted = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            formatted.append(line + "\n" if not line.endswith("\n") else line)
        else:
            formatted.append(line.rstrip("\n"))
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": formatted
    }


def new_code(source):
    """Create a new code cell."""
    if isinstance(source, list):
        lines = source
    else:
        lines = source.split("\n")
    formatted = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            formatted.append(line + "\n" if not line.endswith("\n") else line)
        else:
            formatted.append(line.rstrip("\n"))
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": formatted
    }


def modify_source(cell, old, new):
    """Replace text in a cell's source. Returns modified copy."""
    c = copy.deepcopy(cell)
    src = "".join(c["source"])
    src = src.replace(old, new)
    # Re-split into lines
    lines = src.split("\n")
    formatted = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            formatted.append(line + "\n")
        else:
            formatted.append(line)
    c["source"] = formatted
    return c


def cell_source(cell):
    """Get the full source text of a cell."""
    return "".join(cell["source"])


# ---------------------------------------------------------------------------
# Load all source notebooks
# ---------------------------------------------------------------------------
print("Loading source notebooks...")

NB00 = load_nb("00_llm_as_a_judge.ipynb")
NB01 = load_nb("01_environment_setup.ipynb")
NB02 = load_nb("02_build_rag_pipeline.ipynb")
NB03 = load_nb("03_deepeval_retriever_metrics.ipynb")
NB04 = load_nb("04_deepeval_generator_metrics.ipynb")
NB05 = load_nb("05_deepeval_advanced.ipynb")
NB06 = load_nb("06_ragas_core_metrics.ipynb")
NB07 = load_nb("07_ragas_advanced.ipynb")
NB08 = load_nb("08_faithfulness_hallucination.ipynb")
NB09 = load_nb("09_agentic_rag_eval.ipynb")
NB10 = load_nb("10_end_to_end_pipeline.ipynb")

print(f"  Loaded all 11 notebooks")

os.makedirs(DST, exist_ok=True)

# ===================================================================
# NB00: LLM-as-a-Judge Theory (~30 cells)
# Keep: Parts 1-3 (cells 0-35), cell 42 (bias summary), cells 83-88
# Remove: Parts 4-5 (cells 43-72), Part 6 (cells 73-82)
# ===================================================================
print("\nBuilding NB00: LLM-as-a-Judge Theory...")

nb00 = empty_nb(NB00.get("metadata"))

# Title + Parts 1-3 (cells 0-35)
nb00["cells"].extend(copy_cells(NB00, range(0, 36)))

# Bias summary table (cell 42)
nb00["cells"].append(copy_cell(NB00["cells"][42]))

# Transition: note that framework internals are in NB03/04
nb00["cells"].append(new_md(
    "---\n"
    "\n"
    "> **Note:** Parts 4-6 from the original notebook (DeepEval/RAGAS internals, "
    "customization) are now covered hands-on in Notebooks 03-05, where you'll see "
    "these concepts applied to real RAG pipeline results."
))

# Part 7: Best Practices + Cost (cells 83-87)
nb00["cells"].extend(copy_cells(NB00, range(83, 88)))

# Part 8: Summary (cell 88)
nb00["cells"].append(copy_cell(NB00["cells"][88]))

# Update the title cell to reflect trimmed content
title_cell = nb00["cells"][0]
title_src = cell_source(title_cell)
if "Parts 4-6" not in title_src:
    # Replace the intro to mention streamlined structure
    nb00["cells"][0] = modify_source(
        title_cell,
        "This notebook is **the prerequisite** to every other notebook in this series.",
        "This notebook is **the prerequisite** to every other notebook in this series.\n\n"
        "> **Streamlined for class:** Theory and bias demos are here; "
        "framework deep-dives are in Notebooks 03-05."
    )

save_nb(nb00, "00_llm_as_a_judge.ipynb")


# ===================================================================
# NB01: Environment Setup (~30 cells)
# Trim: merge some redundant verification cells
# ===================================================================
print("\nBuilding NB01: Environment Setup...")

nb01 = empty_nb(NB01.get("metadata"))

# Copy all cells — NB01 is already 41 cells, mostly setup
# We'll keep it largely as-is but trim some verbose verification
# Keep cells 0-38 (core setup), merge 39-40 (summary + next steps)
nb01["cells"].extend(copy_cells(NB01, range(0, 41)))

save_nb(nb01, "01_environment_setup.ipynb")


# ===================================================================
# NB02: Build RAG Pipeline (copy as-is, 36 cells)
# ===================================================================
print("\nBuilding NB02: Build RAG Pipeline...")

nb02 = empty_nb(NB02.get("metadata"))
nb02["cells"].extend(copy_cells(NB02, range(0, 36)))

save_nb(nb02, "02_build_rag_pipeline.ipynb")


# ===================================================================
# NB03: Retriever Metrics — DeepEval + RAGAS (~35 cells)
# Base: NB03 (DeepEval retriever metrics)
# Merge: NB06 cells for RAGAS ContextPrecision/Recall/EntityRecall
# ===================================================================
print("\nBuilding NB03: Retriever Metrics — DeepEval + RAGAS...")

nb03 = empty_nb(NB03.get("metadata"))

# Title — updated to reflect both frameworks
nb03["cells"].append(new_md(
    "# Retriever Metrics — DeepEval + RAGAS Side-by-Side\n"
    "\n"
    "This notebook provides a thorough walkthrough of **retriever evaluation metrics** "
    "from both **DeepEval** and **RAGAS**, run on the same RAG pipeline results from Notebook 02.\n"
    "\n"
    "### What We Cover\n"
    "\n"
    "| Framework | Metric | What It Measures |\n"
    "|-----------|--------|------------------|\n"
    "| DeepEval | ContextualRelevancyMetric | Are retrieved contexts relevant to the query? |\n"
    "| DeepEval | ContextualPrecisionMetric | Are relevant contexts ranked higher? |\n"
    "| DeepEval | ContextualRecallMetric | Do contexts cover all claims in the expected output? |\n"
    "| RAGAS | LLMContextPrecisionWithReference | Are relevant contexts ranked higher? (RAGAS version) |\n"
    "| RAGAS | LLMContextRecall | Do contexts cover all claims in the reference? |\n"
    "| RAGAS | ContextEntityRecall | Do contexts capture key entities from the reference? |"
))

# Setup cell from NB03 (cell 1 — section divider, cell 2 — imports)
nb03["cells"].append(copy_cell(NB03["cells"][1]))

# Combined imports cell: DeepEval + RAGAS
nb03["cells"].append(new_code(
    "import os\n"
    "import json\n"
    "from dotenv import load_dotenv\n"
    "\n"
    "# Load environment\n"
    "dotenv_path = os.path.join(os.path.dirname(os.getcwd()), \".env\")\n"
    "load_dotenv(dotenv_path)\n"
    "\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "\n"
    "# DeepEval imports\n"
    "from deepeval.metrics import (\n"
    "    ContextualRelevancyMetric,\n"
    "    ContextualPrecisionMetric,\n"
    "    ContextualRecallMetric,\n"
    ")\n"
    "from deepeval.test_case import LLMTestCase\n"
    "\n"
    "# RAGAS imports\n"
    "from ragas import SingleTurnSample, EvaluationDataset, evaluate\n"
    "from ragas.metrics import (\n"
    "    LLMContextPrecisionWithReference,\n"
    "    LLMContextRecall,\n"
    "    ContextEntityRecall,\n"
    ")\n"
    "from langchain_openai import ChatOpenAI, OpenAIEmbeddings\n"
    "from ragas.llms import LangchainLLMWrapper\n"
    "from ragas.embeddings import LangchainEmbeddingsWrapper\n"
    "\n"
    "print(\"All imports successful.\")"
))

# Load pipeline results (NB03 cells 4-6: section header, load code, fallback)
nb03["cells"].extend(copy_cells(NB03, [4, 5, 6]))

# Create DeepEval test cases (NB03 cells 7-8)
nb03["cells"].extend(copy_cells(NB03, [7, 8]))

# RAGAS setup cell — create SingleTurnSamples and evaluator_llm
nb03["cells"].append(new_md(
    "---\n"
    "### RAGAS Setup\n"
    "\n"
    "We create RAGAS `SingleTurnSample` objects from the same pipeline results, "
    "and configure the evaluator LLM and embeddings."
))

nb03["cells"].append(new_code(
    "# Configure RAGAS evaluator LLM and embeddings\n"
    "evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=\"gpt-4o-mini\", temperature=0))\n"
    "evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=\"text-embedding-3-small\"))\n"
    "\n"
    "# Create RAGAS SingleTurnSample objects from pipeline results\n"
    "ragas_samples = []\n"
    "for r in pipeline_results:\n"
    "    sample = SingleTurnSample(\n"
    "        user_input=r[\"query\"],\n"
    "        response=r[\"actual_output\"],\n"
    "        retrieved_contexts=r[\"retrieval_context\"],\n"
    "        reference=r.get(\"expected_output\", \"\"),\n"
    "    )\n"
    "    ragas_samples.append(sample)\n"
    "\n"
    "ragas_dataset = EvaluationDataset(samples=ragas_samples)\n"
    "print(f\"Created {len(ragas_samples)} RAGAS samples.\")"
))

# ---- DeepEval Retriever Metrics ----
nb03["cells"].append(new_md(
    "---\n"
    "## DeepEval Retriever Metrics"
))

# ContextualRelevancy (NB03 cells 9-13)
nb03["cells"].extend(copy_cells(NB03, range(9, 14)))

# ContextualPrecision (NB03 cells 14-18) — skip reranking demo (19-20)
nb03["cells"].extend(copy_cells(NB03, range(14, 18)))

# ContextualRecall (NB03 cells 21-25)
nb03["cells"].extend(copy_cells(NB03, range(21, 26)))

# Combined comparison chart (NB03 cells 26-29)
nb03["cells"].extend(copy_cells(NB03, range(26, 30)))

# ---- RAGAS Retriever Metrics ----
nb03["cells"].append(new_md(
    "---\n"
    "## RAGAS Retriever Metrics\n"
    "\n"
    "Now we run the equivalent RAGAS retriever metrics on the **same data**. "
    "RAGAS metrics use `SingleTurnSample` objects and are run via `evaluate()`."
))

# RAGAS ContextPrecision (NB06 cell 24-25) — fix dataset variable name
nb03["cells"].append(copy_cell(NB06["cells"][24]))  # markdown header
nb03["cells"].append(modify_source(copy_cell(NB06["cells"][25]), "eval_dataset", "ragas_dataset"))

# RAGAS ContextRecall (NB06 cells 26-27)
nb03["cells"].append(copy_cell(NB06["cells"][26]))  # markdown header
nb03["cells"].append(modify_source(copy_cell(NB06["cells"][27]), "eval_dataset", "ragas_dataset"))

# RAGAS ContextEntityRecall (NB06 cells 28-29)
nb03["cells"].append(copy_cell(NB06["cells"][28]))  # markdown header
nb03["cells"].append(modify_source(copy_cell(NB06["cells"][29]), "eval_dataset", "ragas_dataset"))

# ---- Side-by-side comparison ----
nb03["cells"].append(new_md(
    "---\n"
    "## Side-by-Side: DeepEval vs RAGAS Retriever Metrics\n"
    "\n"
    "Let's compare the scores from both frameworks on the same test cases."
))

nb03["cells"].append(new_code(
    "# Combine DeepEval and RAGAS retriever scores into one DataFrame\n"
    "# Note: The RAGAS evaluate() calls above stored results in the results objects.\n"
    "# Here we build a comparison table.\n"
    "\n"
    "comparison_data = {\n"
    "    \"Query\": [tc.input[:45] + \"...\" for tc in test_cases],\n"
    "}\n"
    "\n"
    "# Add DeepEval scores (already computed above)\n"
    "comparison_data[\"DE Relevancy\"] = relevancy_scores\n"
    "comparison_data[\"DE Precision\"] = precision_scores\n"
    "comparison_data[\"DE Recall\"] = recall_scores\n"
    "\n"
    "# Add RAGAS scores from the evaluate results\n"
    "# (These come from the RAGAS evaluate() calls above)\n"
    "try:\n"
    "    cp_df = context_precision_result.to_pandas()\n"
    "    comparison_data[\"RAGAS Precision\"] = cp_df.iloc[:, -1].tolist()\n"
    "except:\n"
    "    print(\"RAGAS Context Precision scores not available\")\n"
    "\n"
    "try:\n"
    "    cr_df = context_recall_result.to_pandas()\n"
    "    comparison_data[\"RAGAS Recall\"] = cr_df.iloc[:, -1].tolist()\n"
    "except:\n"
    "    print(\"RAGAS Context Recall scores not available\")\n"
    "\n"
    "try:\n"
    "    er_df = entity_recall_result.to_pandas()\n"
    "    comparison_data[\"RAGAS Entity Recall\"] = er_df.iloc[:, -1].tolist()\n"
    "except:\n"
    "    print(\"RAGAS Entity Recall scores not available\")\n"
    "\n"
    "comparison_df = pd.DataFrame(comparison_data)\n"
    "print(comparison_df.to_string(index=False))\n"
    "print(f\"\\nDeepEval vs RAGAS averages:\")\n"
    "for col in comparison_df.columns[1:]:\n"
    "    if comparison_df[col].dtype in ['float64', 'int64']:\n"
    "        print(f\"  {col}: {comparison_df[col].mean():.3f}\")"
))

# Summary
nb03["cells"].append(new_md(
    "---\n"
    "## Summary\n"
    "\n"
    "In this notebook we:\n"
    "\n"
    "1. Loaded RAG pipeline test results from Notebook 02\n"
    "2. Measured **three DeepEval retriever metrics**: Contextual Relevancy, Precision, and Recall\n"
    "3. Measured **three RAGAS retriever metrics**: Context Precision, Context Recall, and Entity Recall\n"
    "4. Compared scores **side-by-side** across both frameworks\n"
    "\n"
    "### Key Insight\n"
    "Both frameworks measure similar concepts but use different LLM prompting strategies, "
    "which can lead to score differences. Running both gives you a more robust picture.\n"
    "\n"
    "### Next Steps\n"
    "Proceed to **Notebook 04** for generator metrics (Faithfulness, Answer Relevancy) from both frameworks."
))

save_nb(nb03, "03_retriever_metrics.ipynb")


# ===================================================================
# NB04: Generator Metrics — DeepEval + RAGAS (~35 cells)
# Base: NB04 (DeepEval Faithfulness/Relevancy/Hallucination)
# Merge: NB06 cells for RAGAS Faithfulness/Relevancy/FactualCorrectness/SemanticSimilarity
# ===================================================================
print("\nBuilding NB04: Generator Metrics — DeepEval + RAGAS...")

nb04 = empty_nb(NB04.get("metadata"))

# Title
nb04["cells"].append(new_md(
    "# Generator Metrics — DeepEval + RAGAS Side-by-Side\n"
    "\n"
    "This notebook covers **generator evaluation metrics** from both **DeepEval** and **RAGAS**, "
    "measuring whether the LLM's generated answers are faithful, relevant, and correct.\n"
    "\n"
    "### What We Cover\n"
    "\n"
    "| Framework | Metric | What It Measures |\n"
    "|-----------|--------|------------------|\n"
    "| DeepEval | AnswerRelevancyMetric | Is the answer relevant to the question? |\n"
    "| DeepEval | FaithfulnessMetric | Is every claim supported by the context? |\n"
    "| DeepEval | HallucinationMetric | Does the answer contradict the context? |\n"
    "| RAGAS | ResponseRelevancy | Is the answer relevant? (reverse-question approach) |\n"
    "| RAGAS | Faithfulness | Is every claim supported by the context? |\n"
    "| RAGAS | FactualCorrectness | Is the answer factually correct vs reference? |\n"
    "| RAGAS | SemanticSimilarity | How semantically close is the answer to the reference? |"
))

# Setup section
nb04["cells"].append(copy_cell(NB04["cells"][1]))  # --- ## 1. Setup & Imports

# Combined imports cell
nb04["cells"].append(new_code(
    "import os\n"
    "import json\n"
    "from dotenv import load_dotenv\n"
    "\n"
    "dotenv_path = os.path.join(os.path.dirname(os.getcwd()), \".env\")\n"
    "load_dotenv(dotenv_path)\n"
    "\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "\n"
    "# DeepEval imports\n"
    "from deepeval.metrics import (\n"
    "    AnswerRelevancyMetric,\n"
    "    FaithfulnessMetric,\n"
    "    HallucinationMetric,\n"
    "    ContextualRelevancyMetric,\n"
    "    ContextualPrecisionMetric,\n"
    "    ContextualRecallMetric,\n"
    ")\n"
    "from deepeval.test_case import LLMTestCase\n"
    "from deepeval import assert_test\n"
    "\n"
    "# RAGAS imports\n"
    "from ragas import SingleTurnSample, EvaluationDataset, evaluate\n"
    "from ragas.metrics import (\n"
    "    Faithfulness,\n"
    "    ResponseRelevancy,\n"
    "    FactualCorrectness,\n"
    "    SemanticSimilarity,\n"
    ")\n"
    "from langchain_openai import ChatOpenAI, OpenAIEmbeddings\n"
    "from ragas.llms import LangchainLLMWrapper\n"
    "from ragas.embeddings import LangchainEmbeddingsWrapper\n"
    "\n"
    "print(\"All imports successful.\")"
))

# Load pipeline results (NB04 cells 4-7: header, load code, fallback, create test cases)
nb04["cells"].extend(copy_cells(NB04, [4, 5, 6, 7]))

# RAGAS setup
nb04["cells"].append(new_md(
    "---\n"
    "### RAGAS Setup\n"
    "\n"
    "Create RAGAS `SingleTurnSample` objects and configure the evaluator."
))

nb04["cells"].append(new_code(
    "# Configure RAGAS evaluator LLM and embeddings\n"
    "evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=\"gpt-4o-mini\", temperature=0))\n"
    "evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=\"text-embedding-3-small\"))\n"
    "\n"
    "# Create RAGAS SingleTurnSample objects\n"
    "ragas_samples = []\n"
    "for r in pipeline_results:\n"
    "    sample = SingleTurnSample(\n"
    "        user_input=r[\"query\"],\n"
    "        response=r[\"actual_output\"],\n"
    "        retrieved_contexts=r[\"retrieval_context\"],\n"
    "        reference=r.get(\"expected_output\", \"\"),\n"
    "    )\n"
    "    ragas_samples.append(sample)\n"
    "\n"
    "ragas_dataset = EvaluationDataset(samples=ragas_samples)\n"
    "print(f\"Created {len(ragas_samples)} RAGAS samples.\")"
))

# ---- DeepEval Generator Metrics ----
nb04["cells"].append(new_md("---\n## DeepEval Generator Metrics"))

# AnswerRelevancy (NB04 cells 8-13)
nb04["cells"].extend(copy_cells(NB04, range(8, 14)))

# Faithfulness (NB04 cells 14-21, skip 20-21 truths_extraction_limit — deep-dive in NB08)
nb04["cells"].extend(copy_cells(NB04, range(14, 20)))

# Hallucination (NB04 cells 22-25)
nb04["cells"].extend(copy_cells(NB04, range(22, 26)))

# ---- RAGAS Generator Metrics ----
nb04["cells"].append(new_md(
    "---\n"
    "## RAGAS Generator Metrics\n"
    "\n"
    "Now we run the equivalent RAGAS generator metrics on the **same data**."
))

# RAGAS Faithfulness (NB06 cells 19-21) — fix dataset variable name
nb04["cells"].append(copy_cell(NB06["cells"][19]))  # markdown
nb04["cells"].append(modify_source(
    copy_cell(NB06["cells"][20]), "eval_dataset", "ragas_dataset"
))
nb04["cells"].append(copy_cell(NB06["cells"][21]))  # interpreting

# RAGAS ResponseRelevancy (NB06 cells 22-23)
nb04["cells"].append(copy_cell(NB06["cells"][22]))  # markdown
nb04["cells"].append(modify_source(
    copy_cell(NB06["cells"][23]), "eval_dataset", "ragas_dataset"
))

# RAGAS FactualCorrectness + SemanticSimilarity (NB06 cells 30-31)
nb04["cells"].append(copy_cell(NB06["cells"][30]))  # markdown
nb04["cells"].append(modify_source(
    copy_cell(NB06["cells"][31]), "eval_dataset", "ragas_dataset"
))

# ---- Side-by-side comparison ----
nb04["cells"].append(new_md(
    "---\n"
    "## Side-by-Side: DeepEval vs RAGAS Generator Metrics\n"
    "\n"
    "Compare scores from both frameworks on the same test cases."
))

nb04["cells"].append(new_code(
    "# Build comparison table for generator metrics\n"
    "comparison_data = {\n"
    "    \"Query\": [tc.input[:45] + \"...\" for tc in test_cases],\n"
    "    \"DE Relevancy\": ar_scores,\n"
    "    \"DE Faithfulness\": faith_scores,\n"
    "}\n"
    "\n"
    "# Add RAGAS scores\n"
    "try:\n"
    "    faith_df = faithfulness_result.to_pandas()\n"
    "    comparison_data[\"RAGAS Faithfulness\"] = faith_df.iloc[:, -1].tolist()\n"
    "except:\n"
    "    print(\"RAGAS Faithfulness scores not available\")\n"
    "\n"
    "try:\n"
    "    rel_df = relevancy_result.to_pandas()\n"
    "    comparison_data[\"RAGAS Relevancy\"] = rel_df.iloc[:, -1].tolist()\n"
    "except:\n"
    "    print(\"RAGAS Relevancy scores not available\")\n"
    "\n"
    "comparison_df = pd.DataFrame(comparison_data)\n"
    "print(comparison_df.to_string(index=False))\n"
    "\n"
    "print(f\"\\nFramework Averages:\")\n"
    "for col in comparison_df.columns[1:]:\n"
    "    if comparison_df[col].dtype in ['float64', 'int64']:\n"
    "        print(f\"  {col}: {comparison_df[col].mean():.3f}\")"
))

# Summary and interpretation (NB04 cells 32-34) — skip "Run ALL 5" batch (26-31)
nb04["cells"].extend(copy_cells(NB04, range(32, 35)))

# Update the "Next Steps" to reference new NB05
last_cell = nb04["cells"][-1]
if "Notebook 05" in cell_source(last_cell):
    nb04["cells"][-1] = modify_source(
        last_cell,
        "Notebook 05",
        "Notebook 05 (G-Eval & Custom Criteria)"
    )

save_nb(nb04, "04_generator_metrics.ipynb")


# ===================================================================
# NB05: G-Eval & Custom Criteria (~35 cells)
# Source: NB05 cells 0-33 (setup + G-Eval) + cells 46-52 (Custom Metrics)
# ===================================================================
print("\nBuilding NB05: G-Eval & Custom Criteria...")

nb05 = empty_nb(NB05.get("metadata"))

# Title
nb05["cells"].append(new_md(
    "# G-Eval & Custom Criteria\n"
    "\n"
    "This notebook covers DeepEval's most flexible evaluation tools:\n"
    "\n"
    "1. **G-Eval** — Chain-of-thought evaluation with custom criteria, evaluation steps, and rubrics\n"
    "2. **Custom Metrics** — Build your own deterministic and LLM-based metrics\n"
    "\n"
    "### Prerequisites\n"
    "- Notebook 02 (pipeline results in `data/pipeline_results.json`)\n"
    "- Notebook 03-04 (familiarity with DeepEval/RAGAS retriever and generator metrics)"
))

# Setup (NB05 cells 1-6)
nb05["cells"].extend(copy_cells(NB05, range(1, 7)))

# G-Eval section (NB05 cells 7-33)
nb05["cells"].extend(copy_cells(NB05, range(7, 34)))

# Custom Metrics section (NB05 cells 46-52)
nb05["cells"].extend(copy_cells(NB05, range(46, 53)))

# Summary
nb05["cells"].append(new_md(
    "---\n"
    "## Summary\n"
    "\n"
    "### G-Eval\n"
    "- **Criteria-based**: Simple string describing what to evaluate\n"
    "- **Evaluation steps**: Explicit step-by-step instructions for the judge\n"
    "- **Rubric**: Score ranges with expected outcomes for deterministic scoring bands\n"
    "- **Use cases**: Professionalism, PII detection, medical faithfulness, clarity\n"
    "\n"
    "### Custom Metrics\n"
    "- **Deterministic**: ResponseLengthMetric, CitationFormatMetric (no LLM needed)\n"
    "- **LLM-based**: DomainAccuracyMetric (uses OpenAI as judge)\n"
    "\n"
    "### Next Steps\n"
    "- **Notebook 06**: DAG Metric — deterministic decision trees for structured evaluation\n"
    "- **Notebook 07**: Datasets & synthetic test generation"
))

save_nb(nb05, "05_geval_custom_metrics.ipynb")


# ===================================================================
# NB06: DAG & Deterministic Eval (~22 cells)
# Source: NB05 cells 34-45 (DAG section)
# Plus: new setup preamble, exercise, summary
# ===================================================================
print("\nBuilding NB06: DAG & Deterministic Eval...")

nb06 = empty_nb(NB05.get("metadata"))

# Title
nb06["cells"].append(new_md(
    "# DAG Metric — Deterministic Decision Trees for Evaluation\n"
    "\n"
    "The **DAG (Directed Acyclic Graph) Metric** lets you build structured, deterministic "
    "evaluation trees that combine binary/non-binary judgement nodes with optional GEval scoring.\n"
    "\n"
    "### Why DAGs?\n"
    "- **Transparent**: Every decision path is visible and debuggable\n"
    "- **Deterministic**: Same input always follows the same decision tree\n"
    "- **Composable**: Mix rule-based checks with LLM judgements\n"
    "- **Hybrid**: Leaf nodes can delegate to GEval for subjective scoring\n"
    "\n"
    "### Prerequisites\n"
    "- Notebook 05 (familiarity with G-Eval)"
))

# Setup
nb06["cells"].append(new_md("---\n## 1. Setup & Imports"))

nb06["cells"].append(new_code(
    "import os\n"
    "import json\n"
    "from dotenv import load_dotenv\n"
    "\n"
    "dotenv_path = os.path.join(os.path.dirname(os.getcwd()), \".env\")\n"
    "load_dotenv(dotenv_path)\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "from deepeval.test_case import LLMTestCase, LLMTestCaseParams\n"
    "\n"
    "print(\"Setup complete.\")"
))

# Load test data
nb06["cells"].append(new_md("---\n## 2. Load Test Data"))

nb06["cells"].append(new_code(
    "# Load pipeline results from Notebook 02\n"
    "data_path = os.path.join(os.getcwd(), \"data\", \"pipeline_results.json\")\n"
    "\n"
    "with open(data_path) as f:\n"
    "    pipeline_results = json.load(f)\n"
    "\n"
    "# Create test cases\n"
    "test_cases = []\n"
    "for r in pipeline_results:\n"
    "    tc = LLMTestCase(\n"
    "        input=r[\"query\"],\n"
    "        actual_output=r[\"actual_output\"],\n"
    "        expected_output=r.get(\"expected_output\", \"\"),\n"
    "        retrieval_context=r[\"retrieval_context\"],\n"
    "    )\n"
    "    test_cases.append(tc)\n"
    "\n"
    "print(f\"Loaded {len(test_cases)} test cases.\")"
))

# DAG content (NB05 cells 34-45)
nb06["cells"].extend(copy_cells(NB05, range(34, 46)))

# Guided exercise
nb06["cells"].append(new_md(
    "---\n"
    "## Build Your Own DAG — Guided Exercise\n"
    "\n"
    "Try building a DAG for **customer support quality** evaluation:\n"
    "\n"
    "1. **Root node** (Binary): \"Does the response address the customer's question?\"\n"
    "   - No → VerdictNode(score=0)\n"
    "   - Yes → go to step 2\n"
    "2. **Second node** (NonBinary): \"What is the tone of the response?\"\n"
    "   - Professional → VerdictNode(score=1.0)\n"
    "   - Neutral → VerdictNode(score=0.7)\n"
    "   - Unprofessional → VerdictNode(score=0.2)\n"
    "\n"
    "```python\n"
    "# Your code here — define the nodes and build the DAG\n"
    "# tone_professional = VerdictNode(verdict=\"Professional\", score=1.0)\n"
    "# ...\n"
    "```"
))

# Summary
nb06["cells"].append(new_md(
    "---\n"
    "## Summary\n"
    "\n"
    "### DAG Node Types\n"
    "| Node | Purpose | Children |\n"
    "|------|---------|----------|\n"
    "| TaskNode | Preprocessing/extraction | 1 child |\n"
    "| BinaryJudgementNode | Yes/No decision | 2 children (yes/no) |\n"
    "| NonBinaryJudgementNode | Multiple-choice | N children |\n"
    "| VerdictNode | Leaf with score | 0 or 1 GEval child |\n"
    "\n"
    "### GEval vs DAG Decision Guide\n"
    "| Scenario | Use GEval | Use DAG |\n"
    "|----------|-----------|--------|\n"
    "| Subjective quality assessment | ✓ | |\n"
    "| Format/structure checking | | ✓ |\n"
    "| Multi-step decision logic | | ✓ |\n"
    "| Hybrid (rules + subjective) | | ✓ (with GEval child) |\n"
    "\n"
    "### Next Steps\n"
    "- **Notebook 07**: Datasets & synthetic test generation"
))

save_nb(nb06, "06_dag_metric.ipynb")


# ===================================================================
# NB07: Datasets & Synthetic Generation (~30 cells)
# Source: NB05 cells 53-70 (EvaluationDataset + Synthesizer + Pytest)
#         NB07 cells 7-12 (RAGAS TestsetGenerator)
# ===================================================================
print("\nBuilding NB07: Datasets & Synthetic Generation...")

nb07 = empty_nb(NB05.get("metadata"))

# Title
nb07["cells"].append(new_md(
    "# Datasets & Synthetic Test Generation\n"
    "\n"
    "This notebook covers how to create, manage, and automatically generate evaluation datasets "
    "using both **DeepEval** and **RAGAS**.\n"
    "\n"
    "### What We Cover\n"
    "1. **DeepEval EvaluationDataset** — Create from scratch, JSON, CSV\n"
    "2. **DeepEval Synthesizer** — Auto-generate test cases from documents\n"
    "3. **Pytest integration** — Run evaluations in CI/CD\n"
    "4. **RAGAS TestsetGenerator** — Generate test sets from knowledge base\n"
    "5. **Comparison** — DeepEval Synthesizer vs RAGAS TestsetGenerator"
))

# Setup
nb07["cells"].append(new_md("---\n## 1. Setup"))

nb07["cells"].append(new_code(
    "import os\n"
    "import json\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "from dotenv import load_dotenv\n"
    "\n"
    "dotenv_path = os.path.join(os.path.dirname(os.getcwd()), \".env\")\n"
    "load_dotenv(dotenv_path)\n"
    "\n"
    "from deepeval.test_case import LLMTestCase\n"
    "\n"
    "# Load pipeline results\n"
    "data_path = os.path.join(os.getcwd(), \"data\", \"pipeline_results.json\")\n"
    "with open(data_path) as f:\n"
    "    pipeline_results = json.load(f)\n"
    "\n"
    "# Create test cases\n"
    "test_cases = []\n"
    "for r in pipeline_results:\n"
    "    tc = LLMTestCase(\n"
    "        input=r[\"query\"],\n"
    "        actual_output=r[\"actual_output\"],\n"
    "        expected_output=r.get(\"expected_output\", \"\"),\n"
    "        retrieval_context=r[\"retrieval_context\"],\n"
    "    )\n"
    "    test_cases.append(tc)\n"
    "\n"
    "print(f\"Loaded {len(test_cases)} test cases.\")"
))

# DeepEval EvaluationDataset (NB05 cells 53-63)
nb07["cells"].extend(copy_cells(NB05, range(53, 64)))

# DeepEval Synthesizer (NB05 cells 64-70)
nb07["cells"].extend(copy_cells(NB05, range(64, 71)))

# Transition to RAGAS
nb07["cells"].append(new_md(
    "---\n"
    "## RAGAS TestsetGenerator\n"
    "\n"
    "RAGAS provides its own test set generation approach. While DeepEval's Synthesizer "
    "generates individual test cases, RAGAS TestsetGenerator creates complete evaluation "
    "datasets with different query complexity levels."
))

# RAGAS TestsetGenerator (NB07 cells 6-12)
nb07["cells"].extend(copy_cells(NB07, range(6, 13)))

# Comparison table
nb07["cells"].append(new_md(
    "---\n"
    "## DeepEval Synthesizer vs RAGAS TestsetGenerator\n"
    "\n"
    "| Feature | DeepEval Synthesizer | RAGAS TestsetGenerator |\n"
    "|---------|---------------------|----------------------|\n"
    "| Input | List of document strings | LangChain Documents |\n"
    "| Output | List of Golden objects | Testset with DataFrames |\n"
    "| Query types | Configurable scenarios | Simple, Multi-context, Reasoning |\n"
    "| Reference answers | Generated automatically | Generated from documents |\n"
    "| LLM requirement | Any DeepEval-compatible LLM | LangChain-wrapped LLM |\n"
    "| Best for | Quick synthetic datasets | Comprehensive test coverage |\n"
    "\n"
    "**Recommendation:** Use DeepEval Synthesizer for quick iteration during development, "
    "and RAGAS TestsetGenerator for comprehensive evaluation before releases."
))

# Summary
nb07["cells"].append(new_md(
    "---\n"
    "## Summary\n"
    "\n"
    "1. **EvaluationDataset** lets you manage test cases from JSON, CSV, or code\n"
    "2. **Synthesizer** auto-generates test cases from your knowledge base documents\n"
    "3. **Pytest integration** enables CI/CD evaluation pipelines\n"
    "4. **RAGAS TestsetGenerator** provides complementary test generation\n"
    "\n"
    "### Next Steps\n"
    "- **Notebook 08**: Faithfulness deep dive — hallucination types, edge cases, mitigation strategies"
))

save_nb(nb07, "07_datasets_synthetic.ipynb")


# ===================================================================
# NB08: Faithfulness Deep Dive (~25 cells)
# Keep unique content: cells 0-3 (taxonomy), 10-13 (edge cases),
#   16-18 (RAGAS vs DeepEval), 23-38 (strategies, temp, pipeline)
# Remove: 4-9, 14-15, 19-22 (basic intro — already in NB04)
# ===================================================================
print("\nBuilding NB08: Faithfulness Deep Dive...")

nb08 = empty_nb(NB08.get("metadata"))

# Updated title
nb08["cells"].append(new_md(
    "# Deep Dive: Faithfulness, Hallucination & Grounding\n"
    "\n"
    "This notebook is a focused deep-dive on **faithfulness** and **hallucination detection** "
    "in RAG systems. It builds on the basics covered in Notebook 04.\n"
    "\n"
    "### What We Cover\n"
    "1. Hallucination taxonomy (intrinsic, extrinsic, fabrication)\n"
    "2. Edge cases: partial hallucination, ambiguous claims, inference\n"
    "3. RAGAS vs DeepEval faithfulness comparison\n"
    "4. Strategies to reduce hallucination\n"
    "5. Temperature effects on faithfulness\n"
    "6. Building a hallucination detection pipeline\n"
    "\n"
    "> **Prerequisite:** Notebook 04 covers basic Faithfulness and Hallucination metrics."
))

# Setup (NB08 cell 1)
nb08["cells"].append(copy_cell(NB08["cells"][1]))

# Part 1: Hallucination taxonomy (NB08 cells 2-3)
nb08["cells"].extend(copy_cells(NB08, [2, 3]))

# Edge cases (NB08 cells 10-13)
nb08["cells"].append(new_md(
    "---\n"
    "## Edge Cases: Partial Hallucination and Ambiguous Claims\n"
    "\n"
    "These are the hardest cases for faithfulness metrics to handle correctly."
))
nb08["cells"].extend(copy_cells(NB08, range(10, 14)))

# RAGAS vs DeepEval comparison (NB08 cells 16-18)
nb08["cells"].extend(copy_cells(NB08, range(16, 19)))

# Strategies + temperature + pipeline (NB08 cells 23-38)
nb08["cells"].extend(copy_cells(NB08, range(23, 39)))

save_nb(nb08, "08_faithfulness_deep_dive.ipynb")


# ===================================================================
# NB09: Safety & Agentic Eval (~32 cells)
# Keep: NB09 cells 0-21 (agentic RAG)
# Replace: NB09 cells 22-25 (thin safety) with NB05 cells 77-86 (richer safety)
# Keep: NB09 cells 26-31 (tracing)
# ===================================================================
print("\nBuilding NB09: Safety & Agentic Eval...")

nb09 = empty_nb(NB09.get("metadata"))

# Agentic RAG content (NB09 cells 0-21)
nb09["cells"].extend(copy_cells(NB09, range(0, 22)))

# Keep NB09's original safety section (cells 22-25) — integrated with agent
nb09["cells"].extend(copy_cells(NB09, range(22, 26)))

# Add deliberate bias/toxic examples from NB05 (self-contained cells)
nb09["cells"].append(new_md(
    "### Deliberate Bias and Toxicity Examples\n"
    "\n"
    "The cells above tested our agent's normal outputs. Now let's see what "
    "the metrics detect on deliberately problematic responses."
))

# Deliberate bias example (NB05 cell 81)
nb09["cells"].append(copy_cell(NB05["cells"][81]))

# Toxicity section header + deliberate toxic example (NB05 cells 82, 85)
nb09["cells"].append(copy_cell(NB05["cells"][82]))  # ToxicityMetric header
nb09["cells"].append(copy_cell(NB05["cells"][83]))  # ToxicityMetric setup
nb09["cells"].append(copy_cell(NB05["cells"][85]))  # Deliberate toxic example

# Safety summary (NB05 cell 86)
nb09["cells"].append(copy_cell(NB05["cells"][86]))

# Tracing (NB09 cells 26-31)
nb09["cells"].extend(copy_cells(NB09, range(26, 32)))

save_nb(nb09, "09_agentic_rag_eval.ipynb")


# ===================================================================
# NB10: End-to-End Pipeline (~35 cells)
# Replace inline DOCUMENTS with file loading
# Keep rest, trim analysis to hit ~35 cells
# ===================================================================
print("\nBuilding NB10: End-to-End Pipeline...")

nb10 = empty_nb(NB10.get("metadata"))

# Cells 0-3 (title, imports, step 1 header, RAGConfig)
nb10["cells"].extend(copy_cells(NB10, range(0, 4)))

# Replace cell 4 (inline DOCUMENTS) with file loading
nb10["cells"].append(new_code(
    "# Load knowledge base from file (created in Notebook 02)\n"
    "kb_path = os.path.join(os.getcwd(), \"data\", \"knowledge_base.json\")\n"
    "\n"
    "if os.path.exists(kb_path):\n"
    "    with open(kb_path) as f:\n"
    "        DOCUMENTS = json.load(f)\n"
    "    print(f\"Loaded {len(DOCUMENTS)} documents from {kb_path}\")\n"
    "else:\n"
    "    # Fallback: inline documents\n"
    "    print(\"knowledge_base.json not found, using inline documents.\")\n"
    "    DOCUMENTS = [\n"
    "        {\"id\": 1, \"title\": \"Return Policy Overview\", \"content\":\n"
    "         \"Acme Corp offers a 30-day return policy for most items. \"\n"
    "         \"Items must be unused, in original packaging, and accompanied by a receipt. \"\n"
    "         \"Refunds are processed within 5-7 business days.\"},\n"
    "        {\"id\": 2, \"title\": \"Shipping Information\", \"content\":\n"
    "         \"We offer Standard Shipping (5-7 business days, free on orders over $50), \"\n"
    "         \"Expedited Shipping (2-3 business days, $12.99), and \"\n"
    "         \"Overnight Shipping (next business day, $24.99).\"},\n"
    "    ]\n"
    "    print(f\"Using {len(DOCUMENTS)} fallback documents.\")"
))

# Remaining cells 5-40 — keep most, skip some verbose analysis
# Steps 2-4 (cells 5-15)
nb10["cells"].extend(copy_cells(NB10, range(5, 16)))

# Step 5: Analysis (cells 16-24) — keep key cells, trim verbose
nb10["cells"].extend(copy_cells(NB10, range(16, 25)))

# Steps 6-8 (cells 25-40)
nb10["cells"].extend(copy_cells(NB10, range(25, 41)))

save_nb(nb10, "10_end_to_end_pipeline.ipynb")


# ===================================================================
# Copy data directory
# ===================================================================
print("\nCopying data directory...")

src_data = os.path.join(SRC, "data")
dst_data = os.path.join(DST, "data")

if os.path.exists(src_data):
    shutil.copytree(src_data, dst_data, dirs_exist_ok=True)
    print(f"  Copied {src_data} → {dst_data}")


# ===================================================================
# Also save knowledge_base.json for NB10
# ===================================================================
print("\nSaving knowledge_base.json for NB10...")

# Extract DOCUMENTS from NB02 cell 5 if possible, or create from pipeline
kb_path = os.path.join(dst_data, "knowledge_base.json")
if not os.path.exists(kb_path):
    # We'll create a minimal one — the full one is built in NB02
    print("  knowledge_base.json will be created when NB02 runs.")


# ===================================================================
# Final summary
# ===================================================================
print("\n" + "=" * 60)
print("RESTRUCTURE COMPLETE")
print("=" * 60)

for f in sorted(os.listdir(DST)):
    if f.endswith(".ipynb"):
        nb = json.load(open(os.path.join(DST, f)))
        print(f"  {f}: {len(nb['cells'])} cells")

print(f"\nOutput directory: {DST}/")
print("Next: validate notebooks, then swap into workbooks/")
