# RAG Evals — The Complete RAG & Agentic RAG Evaluation Toolkit

A single-source repository for learning, implementing, and mastering RAG evaluation using **DeepEval** and **RAGAS** frameworks. Includes a shared Python module, CLI tool, Streamlit dashboard, golden datasets, evaluation history tracking, and comprehensive learning materials.

## Repository Structure

```
ragevals/
├── ragevals/                        # Python package (pip install -e .)
│   ├── __init__.py                  # Public API exports
│   ├── config.py                    # RAGConfig dataclass + load_env()
│   ├── chunking.py                  # chunk_text(), chunk_documents()
│   ├── embeddings.py                # get_embeddings()
│   ├── vectorstore.py               # Qdrant collection CRUD, build_index()
│   ├── retriever.py                 # retrieve(), rerank()
│   ├── generator.py                 # generate(), DEFAULT_SYSTEM_PROMPT
│   ├── pipeline.py                  # RAGPipeline class
│   ├── evaluation.py                # Unified eval: run_deepeval, run_ragas
│   ├── metrics.py                   # Metric factories for both frameworks
│   ├── datasets.py                  # Built-in datasets + I/O utilities
│   ├── visualization.py             # Charts: bar, heatmap, distribution
│   ├── reports.py                   # Markdown & HTML report generation
│   ├── history.py                   # SQLite evaluation history tracker
│   ├── cli.py                       # Click CLI (evaluate, compare, report)
│   └── __main__.py                  # python -m ragevals support
│
├── dashboard/                       # Streamlit evaluation dashboard
│   ├── app.py                       # Main entry point
│   ├── pages/
│   │   ├── 01_query.py              # Single query evaluation
│   │   ├── 02_batch.py              # Batch dataset evaluation
│   │   ├── 03_compare.py            # Side-by-side config comparison
│   │   ├── 04_history.py            # Run history, trends, regressions
│   │   └── 05_datasets.py           # Browse/upload datasets
│   └── components/
│       ├── metric_card.py           # Score display widget
│       └── sidebar.py               # Config builder sidebar
│
├── data/                            # Golden evaluation datasets
│   ├── knowledge_base.json          # 15 Acme Corp documents
│   ├── golden_basic.json            # 25 standard Q&A pairs
│   ├── golden_adversarial.json      # 15 adversarial edge cases
│   ├── golden_multihop.json         # 10 multi-hop reasoning questions
│   ├── golden_conversational.json   # 5 multi-turn conversations
│   └── schema.json                  # JSON Schema for validation
│
├── configs/                         # Evaluation configuration files
│   ├── eval_basic.yaml              # Basic 3-metric eval
│   ├── eval_full.yaml               # Full metric suite
│   └── eval_retriever.yaml          # Retriever-focused eval
│
├── learning_tutor/                  # Dense learning materials (basic → advanced)
│   ├── 01_what_are_evals.md
│   ├── 02_llm_evals_vs_traditional.md
│   ├── 03_evaluation_approaches.md
│   ├── 04_rag_evaluation_fundamentals.md
│   ├── 05_deepeval_complete_guide.md
│   ├── 06_ragas_complete_guide.md
│   ├── 07_retriever_metrics_deep_dive.md
│   ├── 08_generator_metrics_deep_dive.md
│   ├── 09_agentic_rag_evaluation.md
│   ├── 10_advanced_topics.md
│   └── cheatsheets/
│       ├── metric_selection_flowchart.md
│       ├── deepeval_quickref.md
│       ├── ragas_quickref.md
│       └── evaluation_checklist.md
│
├── workbooks/                       # Hands-on Jupyter notebooks
│   ├── 00_llm_as_a_judge.ipynb
│   ├── 01_environment_setup.ipynb
│   ├── 02_build_rag_pipeline.ipynb
│   ├── 03_deepeval_retriever_metrics.ipynb
│   ├── 04_deepeval_generator_metrics.ipynb
│   ├── 05_deepeval_advanced.ipynb
│   ├── 06_ragas_core_metrics.ipynb
│   ├── 07_ragas_advanced.ipynb
│   ├── 08_faithfulness_hallucination.ipynb
│   ├── 09_agentic_rag_eval.ipynb
│   └── 10_end_to_end_pipeline.ipynb
│
├── integrations/                    # Framework integration examples
│   ├── langchain_example.py
│   ├── llamaindex_example.py
│   └── haystack_example.py
│
├── tests/                           # Test suite
│   ├── test_config.py
│   ├── test_chunking.py
│   ├── test_datasets.py
│   ├── test_history.py
│   └── test_cli.py
│
├── .github/workflows/               # CI/CD
│   ├── test.yml                     # pytest on push/PR
│   └── eval.yml                     # RAG eval on pipeline changes
│
├── Dockerfile
├── docker-compose.yml
├── setup.py
├── requirements.txt
└── .env                             # Your API keys
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/fnusatvik07/rag-evals.git
cd rag-evals
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Add your API keys
echo "OPENAI_API_KEY=sk-..." > .env
```

## Usage

### Python Package

```python
from ragevals import RAGPipeline, RAGConfig, evaluate_pipeline, GOLDEN_TEST_CASES

# Build a pipeline with custom config
config = RAGConfig(chunk_size=500, top_k=5, temperature=0.0)
pipeline = RAGPipeline(config)

# Run a single query
result = pipeline.run("What is the return policy?")
print(result["answer"])

# Evaluate with both DeepEval and RAGAS
results = evaluate_pipeline(pipeline, GOLDEN_TEST_CASES)
print(results["summary"])
```

### CLI

```bash
# Run evaluation
python -m ragevals evaluate --config configs/eval_basic.yaml

# Compare two configs
python -m ragevals compare configs/eval_basic.yaml configs/eval_full.yaml

# Generate a report
python -m ragevals report -i results.csv -o report.html -f html

# Generate synthetic dataset
python -m ragevals generate-dataset --docs data/knowledge_base.json -o data/synthetic.json

# Browse evaluation history
python -m ragevals history show
python -m ragevals history diff <run1> <run2>
python -m ragevals history baseline <run_id>
```

### Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

### Docker

```bash
docker-compose up
# Dashboard: http://localhost:8501
# Jupyter:   http://localhost:8888
```

### Tests

```bash
pytest tests/ -v
```

## Learning Path

### Phase 1: Foundations (learning_tutor 01-03)
What evaluations are, how LLM evals differ from traditional ML, and the three evaluation approaches.

### Phase 2: RAG Evaluation Theory (learning_tutor 04-06)
RAG-specific concepts, then master both DeepEval and RAGAS frameworks.

### Phase 3: Metrics Mastery (learning_tutor 07-08)
Every retriever and generator metric — how they work, when to use them, tradeoffs.

### Phase 4: Advanced Topics (learning_tutor 09-10)
Agentic RAG evaluation, CI/CD integration, production monitoring, custom metrics.

### Phase 5: Hands-On (workbooks 00-10)
Build a real RAG pipeline and evaluate it end-to-end using both frameworks.

## Frameworks Covered

| Framework | Version | Focus |
|-----------|---------|-------|
| **DeepEval** | 3.x | 50+ metrics, pytest-native, agentic eval |
| **RAGAS** | 0.2.x | Research-backed RAG metrics, LangChain integration |

## RAG Pipeline Architecture

```
Documents → Chunk → Embed → Qdrant Index
                                  ↓
Query → Embed → Retrieve → [Rerank] → Generate Answer
                                  ↓
                    Evaluate (DeepEval + RAGAS)
                                  ↓
                    History DB → Reports → Dashboard
```

## Prerequisites

- Python 3.10+
- OpenAI API key
- Optional: Cohere API key (for reranking)
