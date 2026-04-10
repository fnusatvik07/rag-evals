"""Tests for ragevals.cli -- Click CLI commands."""

import json
import os

import pytest
from click.testing import CliRunner

from ragevals.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_config(tmp_path):
    """Write a minimal YAML config file and return its path."""
    import yaml

    config = {
        "chunk_size": 500,
        "top_k": 3,
        "temperature": 0.0,
        "generation_model": "gpt-4o-mini",
    }
    path = tmp_path / "test_config.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


@pytest.fixture
def sample_dataset(tmp_path):
    """Write a minimal JSON dataset and return its path."""
    data = [
        {"query": "What is the return policy?", "reference": "30-day return policy."},
        {"query": "How to contact support?", "reference": "Call 1-800-ACME-HELP."},
    ]
    path = tmp_path / "test_dataset.json"
    path.write_text(json.dumps(data))
    return str(path)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

class TestCLIGroup:
    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "RAG Evals" in result.output

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "ragevals" in result.output


# ---------------------------------------------------------------------------
# history subcommands (no API keys needed)
# ---------------------------------------------------------------------------

class TestHistoryCommands:
    def test_history_show_empty(self, runner, tmp_path):
        """history show should work with an empty DB."""
        db_path = str(tmp_path / "test.db")
        env = os.environ.copy()
        env["RAGEVALS_HISTORY_DB"] = db_path
        result = runner.invoke(main, ["history", "show"], env=env)
        assert result.exit_code == 0
        assert "No runs found" in result.output

    def test_history_show_with_runs(self, runner, tmp_path):
        """history show should list runs after saving."""
        import pandas as pd
        from ragevals.history import EvaluationHistory

        db_path = str(tmp_path / "test.db")
        hist = EvaluationHistory(db_path=db_path)
        scores_df = pd.DataFrame([
            {"test_index": 0, "query": "Q?", "metric_name": "faithfulness", "score": 0.9},
        ])
        hist.save_run(config={"chunk_size": 500}, scores_df=scores_df)
        hist.close()

        env = os.environ.copy()
        env["RAGEVALS_HISTORY_DB"] = db_path
        result = runner.invoke(main, ["history", "show"], env=env)
        assert result.exit_code == 0
        assert "faithfulness" in result.output

    def test_history_help(self, runner):
        result = runner.invoke(main, ["history", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output
        assert "diff" in result.output
        assert "baseline" in result.output


# ---------------------------------------------------------------------------
# report command (no API keys needed)
# ---------------------------------------------------------------------------

class TestReportCommand:
    def test_report_markdown(self, runner, tmp_path):
        """report should generate markdown from a CSV."""
        import pandas as pd

        csv_path = tmp_path / "results.csv"
        df = pd.DataFrame({
            "query": ["Q1", "Q2"],
            "de_faithfulness": [0.85, 0.75],
            "de_answer_relevancy": [0.9, 0.6],
            "latency_ms": [120, 150],
        })
        df.to_csv(csv_path, index=False)

        out_path = tmp_path / "report.md"
        result = runner.invoke(main, [
            "report", "-i", str(csv_path), "-o", str(out_path), "-f", "markdown",
        ])
        assert result.exit_code == 0
        assert out_path.exists()
        content = out_path.read_text()
        assert "Evaluation Report" in content

    def test_report_html(self, runner, tmp_path):
        """report should generate HTML from a CSV."""
        import pandas as pd

        csv_path = tmp_path / "results.csv"
        df = pd.DataFrame({
            "query": ["Q1"],
            "de_faithfulness": [0.85],
            "latency_ms": [100],
        })
        df.to_csv(csv_path, index=False)

        out_path = tmp_path / "report.html"
        result = runner.invoke(main, [
            "report", "-i", str(csv_path), "-o", str(out_path), "-f", "html",
        ])
        assert result.exit_code == 0
        assert out_path.exists()
        content = out_path.read_text()
        assert "<html>" in content.lower()


# ---------------------------------------------------------------------------
# evaluate command (requires API keys -- just test arg parsing)
# ---------------------------------------------------------------------------

class TestEvaluateCommand:
    def test_evaluate_requires_config(self, runner):
        """evaluate should fail without --config."""
        result = runner.invoke(main, ["evaluate"])
        assert result.exit_code != 0
