"""Tests for ragevals.history -- EvaluationHistory SQLite tracker."""

import pandas as pd
import pytest

from ragevals.history import EvaluationHistory


@pytest.fixture
def hist(tmp_path):
    """Create a fresh EvaluationHistory backed by a temp database."""
    db_path = str(tmp_path / "test_history.db")
    h = EvaluationHistory(db_path=db_path)
    yield h
    h.close()


def _make_scores_df(n_cases=3, metrics=None):
    """Create a sample scores DataFrame in the long format expected by save_run."""
    metrics = metrics or ["faithfulness", "answer_relevancy"]
    rows = []
    for i in range(n_cases):
        for m in metrics:
            rows.append({
                "test_index": i,
                "query": f"What is question {i}?",
                "metric_name": m,
                "score": 0.7 + (i * 0.05) + (0.1 if m == "faithfulness" else 0),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# save_run / get_runs
# ---------------------------------------------------------------------------

class TestSaveAndGetRuns:
    def test_save_run_returns_uuid(self, hist):
        df = _make_scores_df()
        run_id = hist.save_run(config={"chunk_size": 500}, scores_df=df)
        assert isinstance(run_id, str)
        assert len(run_id) == 36  # UUID format

    def test_get_runs_returns_saved_run(self, hist):
        df = _make_scores_df()
        run_id = hist.save_run(config={"chunk_size": 500}, scores_df=df)
        runs = hist.get_runs(limit=10)
        assert len(runs) == 1
        assert runs[0]["id"] == run_id
        assert "faithfulness" in runs[0]["scores"]

    def test_multiple_runs_ordered_by_timestamp(self, hist):
        for i in range(3):
            hist.save_run(config={"iteration": i}, scores_df=_make_scores_df())
        runs = hist.get_runs(limit=10)
        assert len(runs) == 3
        assert runs[0]["config"]["iteration"] == 2

    def test_metadata_persisted(self, hist):
        df = _make_scores_df()
        hist.save_run(
            config={"chunk_size": 500},
            scores_df=df,
            metadata={"dataset": "golden", "note": "test run"},
        )
        runs = hist.get_runs()
        assert runs[0]["metadata"]["dataset"] == "golden"

    def test_limit_respected(self, hist):
        for _ in range(5):
            hist.save_run(config={}, scores_df=_make_scores_df())
        runs = hist.get_runs(limit=3)
        assert len(runs) == 3


# ---------------------------------------------------------------------------
# get_run_detail
# ---------------------------------------------------------------------------

class TestGetRunDetail:
    def test_returns_complete_detail(self, hist):
        df = _make_scores_df()
        run_id = hist.save_run(config={"top_k": 3}, scores_df=df)
        detail = hist.get_run_detail(run_id)
        assert detail["id"] == run_id
        assert detail["config"]["top_k"] == 3
        assert len(detail["summary_scores"]) == 2
        assert len(detail["details"]) == 6

    def test_nonexistent_run_raises(self, hist):
        with pytest.raises(ValueError, match="not found"):
            hist.get_run_detail("nonexistent-id")


# ---------------------------------------------------------------------------
# set_baseline / detect_regression
# ---------------------------------------------------------------------------

class TestBaseline:
    def test_set_baseline(self, hist):
        run_id = hist.save_run(config={}, scores_df=_make_scores_df())
        hist.set_baseline(run_id)
        runs = hist.get_runs()
        assert runs[0]["is_baseline"] is True

    def test_only_one_baseline(self, hist):
        id1 = hist.save_run(config={}, scores_df=_make_scores_df())
        id2 = hist.save_run(config={}, scores_df=_make_scores_df())
        hist.set_baseline(id1)
        hist.set_baseline(id2)
        runs = hist.get_runs()
        baselines = [r for r in runs if r["is_baseline"]]
        assert len(baselines) == 1
        assert baselines[0]["id"] == id2

    def test_detect_regression_no_baseline(self, hist):
        result = hist.detect_regression({"faithfulness": 0.9})
        assert result["overall_passed"] is True
        assert result["baseline_id"] is None

    def test_detect_regression_finds_drop(self, hist):
        df = _make_scores_df()
        run_id = hist.save_run(config={}, scores_df=df)
        hist.set_baseline(run_id)
        result = hist.detect_regression(
            {"faithfulness": 0.3, "answer_relevancy": 0.3},
            threshold=0.05,
        )
        assert result["overall_passed"] is False
        assert len(result["regressions"]) > 0

    def test_detect_regression_passes_when_stable(self, hist):
        df = _make_scores_df()
        run_id = hist.save_run(config={}, scores_df=df)
        hist.set_baseline(run_id)
        runs = hist.get_runs()
        baseline_scores = runs[0]["scores"]
        result = hist.detect_regression(baseline_scores, threshold=0.05)
        assert result["overall_passed"] is True


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------

class TestCompareRuns:
    def test_compare_two_runs(self, hist):
        id1 = hist.save_run(config={"v": 1}, scores_df=_make_scores_df())
        id2 = hist.save_run(config={"v": 2}, scores_df=_make_scores_df())
        comp = hist.compare_runs(id1, id2)
        assert comp["run_1"]["id"] == id1
        assert comp["run_2"]["id"] == id2
        assert len(comp["metric_comparison"]) > 0


# ---------------------------------------------------------------------------
# get_trend
# ---------------------------------------------------------------------------

class TestGetTrend:
    def test_trend_returns_time_series(self, hist):
        for _ in range(5):
            hist.save_run(config={}, scores_df=_make_scores_df())
        trend = hist.get_trend("faithfulness", last_n=10)
        assert len(trend) == 5
        assert trend[0]["timestamp"] <= trend[-1]["timestamp"]

    def test_trend_empty_for_unknown_metric(self, hist):
        hist.save_run(config={}, scores_df=_make_scores_df())
        trend = hist.get_trend("nonexistent_metric", last_n=10)
        assert trend == []


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_context_manager(self, tmp_path):
        db_path = str(tmp_path / "ctx_test.db")
        with EvaluationHistory(db_path=db_path) as h:
            h.save_run(config={}, scores_df=_make_scores_df())
            runs = h.get_runs()
            assert len(runs) == 1
