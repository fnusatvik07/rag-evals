"""Unified evaluation interface for DeepEval and RAGAS."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .pipeline import RAGPipeline


def create_deepeval_test_cases(eval_data: list[dict]) -> list:
    """Convert evaluation data to DeepEval LLMTestCase objects.

    Args:
        eval_data: List of dicts with keys: query, response, reference, contexts.

    Returns:
        List of LLMTestCase objects.
    """
    from deepeval.test_case import LLMTestCase

    test_cases = []
    for d in eval_data:
        test_cases.append(LLMTestCase(
            input=d["query"],
            actual_output=d["response"],
            expected_output=d.get("reference", ""),
            retrieval_context=d.get("contexts", []),
        ))
    return test_cases


def create_ragas_dataset(eval_data: list[dict]):
    """Convert evaluation data to a RAGAS EvaluationDataset.

    Args:
        eval_data: List of dicts with keys: query, response, reference, contexts.

    Returns:
        RAGAS EvaluationDataset.
    """
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=d["query"],
            response=d["response"],
            reference=d.get("reference", ""),
            retrieved_contexts=d.get("contexts", []),
        )
        for d in eval_data
    ]
    return EvaluationDataset(samples=samples)


def run_deepeval(
    eval_data: list[dict],
    metrics: dict | None = None,
    model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """Run DeepEval metrics on evaluation data.

    Args:
        eval_data: List of dicts with keys: query, response, reference, contexts.
        metrics: Optional dict of metric_name -> metric instance.
        model: Model for LLM judge.

    Returns:
        DataFrame with a column for each metric score.
    """
    if metrics is None:
        from .metrics import get_deepeval_rag_metrics
        metrics = get_deepeval_rag_metrics(model=model)

    test_cases = create_deepeval_test_cases(eval_data)
    scores = {name: [] for name in metrics}

    for tc in test_cases:
        for name, metric in metrics.items():
            try:
                metric.measure(tc)
                scores[name].append(metric.score)
            except Exception:
                scores[name].append(np.nan)

    df = pd.DataFrame(scores)
    df.insert(0, "query", [d["query"] for d in eval_data])
    return df


def run_ragas(
    eval_data: list[dict],
    metrics: list | None = None,
    model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """Run RAGAS metrics on evaluation data.

    Args:
        eval_data: List of dicts with keys: query, response, reference, contexts.
        metrics: Optional list of RAGAS metric instances.
        model: Model for LLM judge.

    Returns:
        DataFrame with a column for each metric score.
    """
    if metrics is None:
        from .metrics import get_ragas_rag_metrics
        metrics = get_ragas_rag_metrics(model=model)

    from ragas import evaluate as ragas_evaluate

    dataset = create_ragas_dataset(eval_data)
    results = ragas_evaluate(dataset=dataset, metrics=metrics)
    df = results.to_pandas()
    df.insert(0, "query", [d["query"] for d in eval_data])
    return df


def evaluate_pipeline(
    pipeline: "RAGPipeline",
    test_cases: list[dict],
    frameworks: tuple[str, ...] = ("deepeval", "ragas"),
    model: str = "gpt-4o-mini",
) -> dict:
    """Run a pipeline on test cases and evaluate with both frameworks.

    Args:
        pipeline: A RAGPipeline instance.
        test_cases: List of dicts with 'query' and 'reference' keys.
        frameworks: Which frameworks to use ("deepeval", "ragas", or both).
        model: Model for LLM judge.

    Returns:
        Dict with keys: eval_data, deepeval_df, ragas_df, combined_df, summary.
    """
    # Run pipeline on all test cases
    eval_data = []
    for tc in test_cases:
        result = pipeline.run(tc["query"])
        eval_data.append({
            "query": tc["query"],
            "response": result["answer"],
            "reference": tc.get("reference", ""),
            "contexts": result["contexts"],
            "category": tc.get("category", "unknown"),
            "latency_ms": result["latency_ms"],
        })

    output = {"eval_data": eval_data}

    if "deepeval" in frameworks:
        output["deepeval_df"] = run_deepeval(eval_data, model=model)

    if "ragas" in frameworks:
        output["ragas_df"] = run_ragas(eval_data, model=model)

    # Build combined DataFrame
    combined = pd.DataFrame([{
        "query": d["query"],
        "category": d["category"],
        "latency_ms": d["latency_ms"],
    } for d in eval_data])

    if "deepeval_df" in output:
        de_scores = output["deepeval_df"].drop(columns=["query"], errors="ignore")
        de_scores.columns = [f"de_{c}" for c in de_scores.columns]
        combined = pd.concat([combined, de_scores], axis=1)

    if "ragas_df" in output:
        ra_scores = output["ragas_df"].drop(columns=["query"], errors="ignore")
        ra_scores.columns = [f"ragas_{c}" for c in ra_scores.columns]
        combined = pd.concat([combined, ra_scores], axis=1)

    output["combined_df"] = combined

    # Summary statistics
    metric_cols = [c for c in combined.columns
                   if c.startswith("de_") or c.startswith("ragas_")]
    if metric_cols:
        summary = {}
        for col in metric_cols:
            summary[col] = float(combined[col].mean())
        output["summary"] = summary

    return output


def check_regression(
    current_scores: dict,
    baseline_path: str,
    threshold: float = 0.05,
) -> dict:
    """Compare current scores against a baseline and flag regressions.

    Args:
        current_scores: Dict of metric_name -> current mean score.
        baseline_path: Path to baseline JSON file.
        threshold: Maximum allowed decrease before flagging.

    Returns:
        Dict with keys: regressions, improvements, stable, overall_passed.
    """
    with open(baseline_path) as f:
        baseline_data = json.load(f)

    baseline_scores = baseline_data.get("scores", baseline_data)

    result = {
        "regressions": [],
        "improvements": [],
        "stable": [],
        "overall_passed": True,
    }

    for metric, current in current_scores.items():
        if metric not in baseline_scores:
            continue
        baseline_val = baseline_scores[metric]
        diff = current - baseline_val

        entry = {
            "metric": metric,
            "baseline": baseline_val,
            "current": current,
            "change": diff,
        }

        if diff < -threshold:
            result["regressions"].append(entry)
            result["overall_passed"] = False
        elif diff > threshold:
            result["improvements"].append(entry)
        else:
            result["stable"].append(entry)

    return result
