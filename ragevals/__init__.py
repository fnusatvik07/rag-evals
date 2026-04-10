"""RAG Evaluation Toolkit — Unified evaluation for RAG pipelines using DeepEval and RAGAS."""

__version__ = "0.1.0"

from .config import RAGConfig, load_env
from .history import EvaluationHistory

# Optional imports — these modules may not exist yet during early development.
try:
    from .pipeline import RAGPipeline
except ImportError:
    RAGPipeline = None

try:
    from .evaluation import evaluate_pipeline, run_deepeval, run_ragas
except ImportError:
    evaluate_pipeline = run_deepeval = run_ragas = None

try:
    from .datasets import (
        load_dataset,
        GOLDEN_TEST_CASES,
        generate_synthetic_dataset,
    )
except ImportError:
    load_dataset = GOLDEN_TEST_CASES = generate_synthetic_dataset = None

__all__ = [
    "RAGConfig", "load_env", "RAGPipeline",
    "evaluate_pipeline", "run_deepeval", "run_ragas",
    "load_dataset", "GOLDEN_TEST_CASES",
    "generate_synthetic_dataset",
    "EvaluationHistory",
]
