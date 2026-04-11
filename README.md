<p align="center">
  <h1 align="center">RAG Evals</h1>
  <p align="center">
    <strong>The Complete RAG & Agentic RAG Evaluation Toolkit</strong>
  </p>
  <p align="center">
    11 hands-on notebooks &bull; DeepEval + RAGAS side-by-side &bull; Classroom-ready
  </p>
</p>

<p align="center">
  <a href="#notebooks">Notebooks</a> &nbsp;&middot;&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;&middot;&nbsp;
  <a href="#environment-variables">Env Variables</a> &nbsp;&middot;&nbsp;
  <a href="#learning-path">Learning Path</a> &nbsp;&middot;&nbsp;
  <a href="#architecture">Architecture</a>
</p>

## What Is This?

A single repository for **learning, implementing, and mastering RAG evaluation** using the two leading frameworks:

| | DeepEval | RAGAS |
|---|---|---|
| **Version** | 3.9+ | 0.4+ |
| **Strength** | 50+ metrics, pytest-native, agentic eval | Research-backed RAG metrics, LangChain integration |
| **In This Repo** | Notebooks 03-10 | Notebooks 03-04, 07, 10 |

Both frameworks are taught **side-by-side** so you can compare scores on the same data, understand where they agree and diverge, and pick the right tool for your use case.

## Notebooks

Each notebook is scoped to **one focused topic**, sized for a single class session (~25-40 cells).

| # | Notebook | Topic | Key Concepts |
|---|----------|-------|-------------|
| 00 | `00_llm_as_a_judge.ipynb` | LLM-as-a-Judge Theory | Evaluation paradigms, bias types, cost analysis |
| 01 | `01_environment_setup.ipynb` | Environment Setup | API keys, dependencies, verification |
| 02 | `02_build_rag_pipeline.ipynb` | Build RAG Pipeline | Chunking, embeddings, Qdrant, generation. Saves `pipeline_results.json` |
| 03 | `03_retriever_metrics.ipynb` | Retriever Metrics | ContextualPrecision, Recall, Relevancy (DeepEval + RAGAS) |
| 04 | `04_generator_metrics.ipynb` | Generator Metrics | Faithfulness, AnswerRelevancy, Hallucination (DeepEval + RAGAS) |
| 05 | `05_geval_custom_metrics.ipynb` | G-Eval & Custom Criteria | G-Eval 3.1, custom evaluation steps, threshold tuning |
| 06 | `06_dag_metric.ipynb` | DAG Metric | Deterministic decision-tree evaluation, no LLM calls |
| 07 | `07_datasets_synthetic.ipynb` | Datasets & Synthetic Gen | EvaluationDataset, Synthesizer, RAGAS TestsetGenerator, pytest |
| 08 | `08_faithfulness_deep_dive.ipynb` | Faithfulness Deep Dive | Hallucination taxonomy, edge cases, mitigation strategies, detection pipeline |
| 09 | `09_agentic_rag_eval.ipynb` | Agentic RAG Evaluation | Tool use, multi-turn conversations, agent quality metrics |
| 10 | `10_end_to_end_pipeline.ipynb` | End-to-End Pipeline | Full metrics suite, hyperparameter experiments, regression testing, reporting |

**Data flow**: NB02 builds the RAG pipeline and saves results to `workbooks/data/pipeline_results.json`. Notebooks 03-09 load that file instead of rebuilding the pipeline, so each one runs independently.

## Quick Start

### Prerequisites

- Python 3.13+
- An OpenAI API key
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
# Clone
git clone https://github.com/fnusatvik07/rag-evals.git
cd rag-evals

# Create environment and install dependencies
uv sync

# Add your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run Notebooks

```bash
# Start Jupyter
uv run jupyter lab workbooks/

# Or run a notebook end-to-end from the command line
uv run jupyter nbconvert --to notebook --execute workbooks/02_build_rag_pipeline.ipynb
```

### With pip (alternative)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
jupyter lab workbooks/
```

## Environment Variables

Create a `.env` file in the project root (see `.env.example`):

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | **Yes** | OpenAI API key. Used by all notebooks for embeddings (`text-embedding-3-small`) and generation (`gpt-4o-mini`). |
| `CONFIDENT_API_KEY` | No | DeepEval's [Confident AI](https://app.confident-ai.com) platform key. Enables cloud dashboard, metric tracking, and team collaboration. |
| `LANGCHAIN_API_KEY` | No | [LangSmith](https://smith.langchain.com) API key. Enables tracing for RAGAS metric calls (useful for debugging). |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` to enable LangSmith tracing. Only works with `LANGCHAIN_API_KEY`. |
| `RAGEVALS_HISTORY_DB` | No | Override the default SQLite history database path (`~/.ragevals/history.db`). |

**Cost estimate**: Running all 11 notebooks end-to-end uses approximately $1-3 of OpenAI API credits (mostly `gpt-4o-mini` calls).

## Learning Path

The repository supports two learning tracks:

### Track A: Reading Materials (learning_tutor/)

Progressive markdown guides from foundations to advanced topics:

```
01 What Are Evals?           --> 02 LLM vs Traditional Evals
03 Evaluation Approaches     --> 04 RAG Evaluation Fundamentals
05 DeepEval Complete Guide   --> 06 RAGAS Complete Guide
07 Retriever Metrics Deep Dive --> 08 Generator Metrics Deep Dive
09 Agentic RAG Evaluation    --> 10 Advanced Topics (CI/CD, production)
```

### Track B: Hands-On Notebooks (workbooks/)

```
00 Theory       --> 01 Setup      --> 02 Build Pipeline
                                        |
03 Retriever Metrics  <-- loads pipeline_results.json
04 Generator Metrics  <-- loads pipeline_results.json
05 G-Eval & Custom    <-- loads pipeline_results.json
06 DAG Metric         <-- loads pipeline_results.json
07 Datasets & Synth   <-- loads pipeline_results.json
08 Faithfulness Deep Dive
09 Agentic RAG
10 End-to-End Pipeline (self-contained)
```

## Architecture

```
                    +------------------+
                    |  Knowledge Base  |
                    | (JSON documents) |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   Chunk + Embed  |
                    |  (OpenAI embed)  |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   Qdrant Index   |
                    |   (in-memory)    |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
     +--------v---------+         +--------v---------+
     |     Retrieve      |         |     Generate     |
     | (top-k + rerank)  |         |   (gpt-4o-mini)  |
     +--------+----------+         +--------+---------+
              |                             |
              +-------------+---------------+
                            |
              +-------------v--------------+
              |        Evaluate            |
              |  DeepEval    |    RAGAS    |
              +---+----------+--------+---+
                  |                    |
        +---------v------+    +-------v---------+
        | Retriever       |    | Generator       |
        | - Ctx Precision |    | - Faithfulness  |
        | - Ctx Recall    |    | - Relevancy     |
        | - Ctx Relevancy |    | - Hallucination |
        +-----------------+    +-----------------+
```

## Repository Structure

```
rag-evals/
├── workbooks/                     # 11 Jupyter notebooks (classroom-ready)
│   ├── 00_llm_as_a_judge.ipynb
│   ├── 01_environment_setup.ipynb
│   ├── 02_build_rag_pipeline.ipynb
│   ├── 03_retriever_metrics.ipynb
│   ├── 04_generator_metrics.ipynb
│   ├── 05_geval_custom_metrics.ipynb
│   ├── 06_dag_metric.ipynb
│   ├── 07_datasets_synthetic.ipynb
│   ├── 08_faithfulness_deep_dive.ipynb
│   ├── 09_agentic_rag_eval.ipynb
│   ├── 10_end_to_end_pipeline.ipynb
│   └── data/
│       ├── pipeline_results.json  # Shared RAG outputs (from NB02)
│       ├── knowledge_base.json    # 15 Acme Corp documents
│       └── test_cases.json        # 16 golden Q&A pairs
│
├── ragevals/                      # Python package
│   ├── config.py                  # RAGConfig dataclass + load_env()
│   ├── chunking.py                # chunk_text(), chunk_documents()
│   ├── embeddings.py              # get_embeddings()
│   ├── vectorstore.py             # Qdrant CRUD, build_index()
│   ├── retriever.py               # retrieve(), rerank()
│   ├── generator.py               # generate(), system prompts
│   ├── pipeline.py                # RAGPipeline class
│   ├── evaluation.py              # run_deepeval(), run_ragas()
│   ├── metrics.py                 # Metric factories
│   ├── datasets.py                # Built-in datasets + I/O
│   ├── visualization.py           # Charts
│   ├── reports.py                 # Markdown/HTML reports
│   ├── history.py                 # SQLite eval history
│   └── cli.py                     # CLI (evaluate, compare, report)
│
├── learning_tutor/                # Reading materials (01-10 + cheatsheets)
├── integrations/                  # LangChain, LlamaIndex, Haystack examples
├── tests/                         # pytest suite
├── dashboard/                     # Streamlit evaluation dashboard
├── data/                          # Golden datasets (basic, adversarial, multi-hop)
├── configs/                       # YAML evaluation configs
│
├── .env.example                   # Template for environment variables
├── pyproject.toml                 # Project dependencies (uv / pip)
├── restructure_notebooks.py       # Script used to restructure notebooks
└── README.md
```

## CLI Usage

```bash
# Run evaluation
python -m ragevals evaluate --config configs/eval_basic.yaml

# Compare two configs
python -m ragevals compare configs/eval_basic.yaml configs/eval_full.yaml

# Generate report
python -m ragevals report -i results.csv -o report.html -f html

# Generate synthetic dataset
python -m ragevals generate-dataset --docs data/knowledge_base.json -o data/synthetic.json

# Evaluation history
python -m ragevals history show
python -m ragevals history diff <run1> <run2>
```

## Dashboard

```bash
streamlit run dashboard/app.py
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `deepeval` >= 3.9.6 | Primary evaluation framework (50+ metrics) |
| `ragas` >= 0.4.3 | Research-backed RAG metrics |
| `openai` >= 2.31 | Embeddings + generation |
| `qdrant-client` >= 1.17 | Vector store (in-memory) |
| `langchain-openai` >= 1.1 | LangChain wrappers for RAGAS |
| `sentence-transformers` >= 5.4 | Cross-encoder reranking (local, no API key) |
| `pandas`, `matplotlib`, `numpy` | Data analysis and visualization |

## License

MIT
