"""Command-line interface for RAG Evals.

Provides commands for running evaluations, comparing configs, generating
reports and datasets, and browsing evaluation history.

Usage::

    python -m ragevals evaluate --config configs/eval_basic.yaml
    python -m ragevals history show
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from ragevals.history import EvaluationHistory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str):
    """Load a RAGConfig from a YAML file."""
    import yaml
    from ragevals.config import RAGConfig

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return RAGConfig.from_dict(data)


def _load_test_cases(path: str) -> list[dict]:
    """Load test cases from a JSON or CSV file."""
    import pandas as pd

    p = Path(path)
    if p.suffix == ".json":
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
        return df.to_dict("records")
    else:
        raise click.BadParameter(f"Unsupported file format: {p.suffix}")


def _format_table(rows: list[dict], columns: list[str]) -> str:
    """Render a simple ASCII table from a list of dicts."""
    widths = {c: len(c) for c in columns}
    for row in rows:
        for c in columns:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))

    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    lines = [header, sep]
    for row in rows:
        lines.append(" | ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))
    return "\n".join(lines)


def _scores_to_long(combined_df, metric_cols: list[str]):
    """Reshape a wide combined_df into the long format expected by history.save_run."""
    import pandas as pd

    rows = []
    for idx, row in combined_df.iterrows():
        query = row.get("query", f"Query {idx}")
        for metric in metric_cols:
            rows.append({
                "test_index": idx,
                "query": query,
                "metric_name": metric,
                "score": float(row[metric]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="ragevals", prog_name="ragevals")
def main():
    """RAG Evals -- evaluate, compare, and track RAG pipelines."""


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "-c", "--config", "config_path", required=True,
    type=click.Path(exists=True), help="Path to YAML configuration file.",
)
@click.option(
    "-d", "--dataset", "dataset_path", default=None,
    type=click.Path(exists=True), help="Path to evaluation dataset (JSON/CSV).",
)
@click.option(
    "-o", "--output", "output_path", default=None,
    type=click.Path(), help="Path to save results CSV.",
)
@click.option(
    "--framework", type=click.Choice(["deepeval", "ragas", "both"]), default="both",
    help="Evaluation framework to use.",
)
@click.option(
    "--save-history/--no-history", default=True,
    help="Whether to save this run to the history database.",
)
def evaluate(config_path, dataset_path, output_path, framework, save_history):
    """Run an evaluation using a YAML config."""
    from ragevals.config import load_env
    from ragevals.pipeline import RAGPipeline
    from ragevals.evaluation import evaluate_pipeline
    from ragevals.datasets import GOLDEN_TEST_CASES

    load_env()
    config = _load_config(config_path)
    click.echo(click.style("Configuration: ", fg="cyan") + config.name)

    # Load test cases.
    if dataset_path:
        test_cases = _load_test_cases(dataset_path)
        click.echo(f"Dataset: {dataset_path} ({len(test_cases)} cases)")
    else:
        test_cases = GOLDEN_TEST_CASES
        click.echo(f"Using built-in golden test cases ({len(test_cases)} cases)")

    # Determine frameworks.
    frameworks = ("deepeval", "ragas") if framework == "both" else (framework,)

    click.echo(click.style("Building pipeline...", fg="yellow"))
    pipeline = RAGPipeline(config)

    click.echo(click.style("Running evaluation...", fg="yellow"))
    results = evaluate_pipeline(pipeline, test_cases, frameworks=frameworks)

    # Display summary.
    if "summary" in results:
        click.echo("\n" + click.style("Results:", bold=True))
        rows = [
            {"Metric": k, "Score": f"{v:.4f}", "Status": "PASS" if v >= 0.7 else "FAIL"}
            for k, v in sorted(results["summary"].items())
        ]
        click.echo(_format_table(rows, ["Metric", "Score", "Status"]))

    # Save CSV output.
    if output_path and "combined_df" in results:
        results["combined_df"].to_csv(output_path, index=False)
        click.echo(click.style(f"\nResults saved to {output_path}", fg="green"))

    # Save to history.
    if save_history and "combined_df" in results:
        combined = results["combined_df"]
        metric_cols = [c for c in combined.columns if c.startswith("de_") or c.startswith("ragas_")]

        if metric_cols:
            scores_long = _scores_to_long(combined, metric_cols)
            hist = EvaluationHistory()
            run_id = hist.save_run(
                config=vars(config) if hasattr(config, "__dataclass_fields__") else {},
                scores_df=scores_long,
                metadata={"dataset": dataset_path or "golden", "config_path": config_path},
            )
            click.echo(click.style("Run saved to history: ", fg="green") + run_id)
            hist.close()


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

@main.command()
@click.argument("config1", type=click.Path(exists=True))
@click.argument("config2", type=click.Path(exists=True))
@click.option("-d", "--dataset", "dataset_path", default=None, type=click.Path(exists=True))
@click.option("--framework", type=click.Choice(["deepeval", "ragas", "both"]), default="both")
def compare(config1, config2, dataset_path, framework):
    """Compare two configurations side-by-side."""
    from ragevals.config import load_env
    from ragevals.pipeline import RAGPipeline
    from ragevals.evaluation import evaluate_pipeline
    from ragevals.datasets import GOLDEN_TEST_CASES

    load_env()
    cfg1 = _load_config(config1)
    cfg2 = _load_config(config2)

    test_cases = _load_test_cases(dataset_path) if dataset_path else GOLDEN_TEST_CASES
    frameworks = ("deepeval", "ragas") if framework == "both" else (framework,)

    click.echo(click.style("Config 1: ", fg="cyan") + cfg1.name)
    click.echo(click.style("Config 2: ", fg="cyan") + cfg2.name)

    click.echo(click.style("Evaluating Config 1...", fg="yellow"))
    p1 = RAGPipeline(cfg1)
    r1 = evaluate_pipeline(p1, test_cases, frameworks=frameworks)

    click.echo(click.style("Evaluating Config 2...", fg="yellow"))
    p2 = RAGPipeline(cfg2)
    r2 = evaluate_pipeline(p2, test_cases, frameworks=frameworks)

    s1 = r1.get("summary", {})
    s2 = r2.get("summary", {})

    all_metrics = sorted(set(s1) | set(s2))
    rows = []
    for m in all_metrics:
        v1 = s1.get(m, 0.0)
        v2 = s2.get(m, 0.0)
        delta = v2 - v1
        if delta > 0.01:
            winner = "Config2"
        elif delta < -0.01:
            winner = "Config1"
        else:
            winner = "Tie"
        rows.append({
            "Metric": m,
            "Config1": f"{v1:.4f}",
            "Config2": f"{v2:.4f}",
            "Delta": f"{delta:+.4f}",
            "Winner": winner,
        })

    click.echo("\n" + click.style("Comparison:", bold=True))
    click.echo(_format_table(rows, ["Metric", "Config1", "Config2", "Delta", "Winner"]))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@main.command()
@click.option("-i", "--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", required=True, type=click.Path())
@click.option("-f", "--format", "fmt", type=click.Choice(["markdown", "html"]), default="markdown")
def report(input_path, output_path, fmt):
    """Generate a report from evaluation results CSV."""
    import pandas as pd
    from ragevals.reports import generate_markdown_report, generate_html_report

    df = pd.read_csv(input_path)
    metric_cols = [c for c in df.columns if c.startswith("de_") or c.startswith("ragas_")]

    if not metric_cols:
        metric_cols = [c for c in df.select_dtypes(include="number").columns if c != "latency_ms"]

    if fmt == "markdown":
        content = generate_markdown_report(df, metric_cols)
    else:
        content = generate_html_report(df, metric_cols)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    click.echo(click.style(f"Report written to {output_path}", fg="green"))


# ---------------------------------------------------------------------------
# generate-dataset
# ---------------------------------------------------------------------------

@main.command("generate-dataset")
@click.option(
    "--docs", required=True, type=click.Path(exists=True),
    help="Path to knowledge base JSON.",
)
@click.option("-o", "--output", required=True, type=click.Path(), help="Output JSON path.")
@click.option("--n-per-doc", default=3, type=int, help="Test cases per document.")
def generate_dataset(docs, output, n_per_doc):
    """Generate a synthetic evaluation dataset from a knowledge base."""
    from ragevals.config import load_env
    from ragevals.datasets import load_dataset, save_dataset

    load_env()
    click.echo(click.style("Loading knowledge base...", fg="yellow"))
    kb = load_dataset(docs)

    click.echo(click.style(f"Generating {n_per_doc} test cases per document...", fg="yellow"))

    from openai import OpenAI
    client = OpenAI()

    test_cases = []
    for doc in kb:
        title = doc.get("title", "Unknown")
        content = doc.get("content", "")
        if not content.strip():
            continue

        prompt = (
            f"Given this document about '{title}':\n\n{content}\n\n"
            f"Generate {n_per_doc} question-answer pairs for evaluating a RAG system. "
            f"Return a JSON array where each element has 'query' and 'reference' keys. "
            f"The reference should be a concise factual answer based on the document."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            pairs = data if isinstance(data, list) else data.get("pairs", data.get("questions", []))
            for pair in pairs:
                pair.setdefault("category", title.lower().replace(" ", "_"))
            test_cases.extend(pairs)
            click.echo(f"  {title}: {len(pairs)} cases")
        except Exception as exc:
            click.echo(click.style(f"  {title}: failed ({exc})", fg="red"))

    save_dataset(test_cases, output)
    click.echo(click.style(f"\nDataset written to {output} ({len(test_cases)} cases)", fg="green"))


# ---------------------------------------------------------------------------
# history sub-group
# ---------------------------------------------------------------------------

@main.group()
def history():
    """Browse and manage evaluation history."""


@history.command("show")
@click.option("-n", "--limit", default=20, type=int, help="Number of recent runs to show.")
def history_show(limit):
    """List recent evaluation runs."""
    hist = EvaluationHistory()
    runs = hist.get_runs(limit=limit)
    hist.close()

    if not runs:
        click.echo("No runs found.")
        return

    rows = []
    for r in runs:
        score_str = ", ".join(f"{k}={v:.3f}" for k, v in r["scores"].items())
        rows.append({
            "ID": r["id"][:8] + "...",
            "Timestamp": r["timestamp"][:19],
            "Baseline": "*" if r["is_baseline"] else "",
            "Scores": score_str[:60],
        })

    click.echo(_format_table(rows, ["ID", "Timestamp", "Baseline", "Scores"]))


@history.command("diff")
@click.argument("run1")
@click.argument("run2")
def history_diff(run1, run2):
    """Compare two evaluation runs."""
    hist = EvaluationHistory()

    # Support short IDs by prefix-matching.
    all_runs = hist.get_runs(limit=1000)
    full_1, full_2 = None, None
    for r in all_runs:
        if r["id"].startswith(run1):
            full_1 = r["id"]
        if r["id"].startswith(run2):
            full_2 = r["id"]

    if not full_1 or not full_2:
        click.echo(click.style("Error: Could not resolve run IDs.", fg="red"))
        hist.close()
        sys.exit(1)

    result = hist.compare_runs(full_1, full_2)
    hist.close()

    rows = []
    for c in result["metric_comparison"]:
        rows.append({
            "Metric": c["metric_name"],
            "Run 1": f"{c['score_1']:.4f}",
            "Run 2": f"{c['score_2']:.4f}",
            "Delta": f"{c['delta']:+.4f}",
        })

    click.echo(_format_table(rows, ["Metric", "Run 1", "Run 2", "Delta"]))


@history.command("baseline")
@click.argument("run_id")
def history_baseline(run_id):
    """Set a run as the evaluation baseline."""
    hist = EvaluationHistory()

    # Support short IDs.
    all_runs = hist.get_runs(limit=1000)
    full_id = None
    for r in all_runs:
        if r["id"].startswith(run_id):
            full_id = r["id"]
            break

    if not full_id:
        click.echo(click.style("Error: Run ID not found.", fg="red"))
        hist.close()
        sys.exit(1)

    hist.set_baseline(full_id)
    hist.close()
    click.echo(click.style(f"Baseline set to {full_id}", fg="green"))
