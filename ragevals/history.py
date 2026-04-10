"""SQLite-backed evaluation history tracker.

Stores evaluation runs, scores, and per-query details for trend analysis,
regression detection, and comparison across configurations.

Default database location: ``~/.ragevals/history.db``
Override with the ``RAGEVALS_HISTORY_DB`` environment variable.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_DEFAULT_DB_DIR = os.path.join(str(Path.home()), ".ragevals")
_DEFAULT_DB_PATH = os.path.join(_DEFAULT_DB_DIR, "history.db")


class EvaluationHistory:
    """Persistent store for evaluation run history.

    Parameters
    ----------
    db_path : str or None
        Path to the SQLite database file.  Falls back to the
        ``RAGEVALS_HISTORY_DB`` environment variable, then to
        ``~/.ragevals/history.db``.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = (
            db_path
            or os.environ.get("RAGEVALS_HISTORY_DB")
            or _DEFAULT_DB_PATH
        )
        # Ensure the parent directory exists.
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent read performance.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the required tables if they do not already exist."""
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id           TEXT PRIMARY KEY,
                timestamp    TEXT NOT NULL,
                config_json  TEXT NOT NULL,
                metadata_json TEXT,
                is_baseline  INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS run_scores (
                run_id       TEXT NOT NULL,
                metric_name  TEXT NOT NULL,
                mean_score   REAL NOT NULL,
                min_score    REAL,
                max_score    REAL,
                std_score    REAL,
                PRIMARY KEY (run_id, metric_name),
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS run_details (
                run_id       TEXT NOT NULL,
                test_index   INTEGER NOT NULL,
                query        TEXT,
                metric_name  TEXT NOT NULL,
                score        REAL NOT NULL,
                reason       TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_run_details_run
                ON run_details(run_id);
            CREATE INDEX IF NOT EXISTS idx_run_scores_run
                ON run_scores(run_id);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_run(
        self,
        config: dict,
        scores_df,
        metadata: Optional[dict] = None,
    ) -> str:
        """Persist an evaluation run.

        Parameters
        ----------
        config : dict
            The RAGConfig (or equivalent) used for this run, as a dict.
        scores_df : pandas.DataFrame
            DataFrame with columns: ``test_index``, ``query``,
            ``metric_name``, ``score``, and optionally ``reason``.
        metadata : dict, optional
            Arbitrary metadata to store alongside the run.

        Returns
        -------
        str
            The generated UUID for this run.
        """
        import numpy as np

        run_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()

        cur = self._conn.cursor()

        # Insert the run record.
        cur.execute(
            "INSERT INTO runs (id, timestamp, config_json, metadata_json, is_baseline) "
            "VALUES (?, ?, ?, ?, 0)",
            (run_id, ts, json.dumps(config), json.dumps(metadata or {})),
        )

        # Aggregate per-metric summary scores.
        for metric_name, group in scores_df.groupby("metric_name"):
            scores = group["score"].astype(float)
            cur.execute(
                "INSERT INTO run_scores (run_id, metric_name, mean_score, min_score, max_score, std_score) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    str(metric_name),
                    float(scores.mean()),
                    float(scores.min()),
                    float(scores.max()),
                    float(scores.std()) if len(scores) > 1 else 0.0,
                ),
            )

        # Insert per-query detail rows.
        for _, row in scores_df.iterrows():
            cur.execute(
                "INSERT INTO run_details (run_id, test_index, query, metric_name, score, reason) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    int(row.get("test_index", 0)),
                    str(row.get("query", "")),
                    str(row["metric_name"]),
                    float(row["score"]),
                    str(row.get("reason", "")),
                ),
            )

        self._conn.commit()
        return run_id

    def set_baseline(self, run_id: str) -> None:
        """Mark *run_id* as the baseline, clearing any previous baseline."""
        cur = self._conn.cursor()
        cur.execute("UPDATE runs SET is_baseline = 0 WHERE is_baseline = 1")
        cur.execute("UPDATE runs SET is_baseline = 1 WHERE id = ?", (run_id,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_runs(self, limit: int = 20) -> list[dict]:
        """Return the most recent runs with their summary scores.

        Parameters
        ----------
        limit : int
            Maximum number of runs to return (most recent first).

        Returns
        -------
        list[dict]
            Each dict contains ``id``, ``timestamp``, ``config``,
            ``metadata``, ``is_baseline``, and ``scores`` (a dict mapping
            metric name to mean score).
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, timestamp, config_json, metadata_json, is_baseline "
            "FROM runs ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        runs = []
        for row in cur.fetchall():
            run_id = row["id"]
            # Fetch summary scores for this run.
            score_cur = self._conn.cursor()
            score_cur.execute(
                "SELECT metric_name, mean_score FROM run_scores WHERE run_id = ?",
                (run_id,),
            )
            scores = {r["metric_name"]: r["mean_score"] for r in score_cur.fetchall()}
            runs.append(
                {
                    "id": run_id,
                    "timestamp": row["timestamp"],
                    "config": json.loads(row["config_json"]),
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                    "is_baseline": bool(row["is_baseline"]),
                    "scores": scores,
                }
            )
        return runs

    def get_run_detail(self, run_id: str) -> dict:
        """Return full detail for a single run.

        Returns
        -------
        dict
            Keys: ``id``, ``timestamp``, ``config``, ``metadata``,
            ``is_baseline``, ``summary_scores`` (list of dicts), and
            ``details`` (list of per-query dicts).
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, timestamp, config_json, metadata_json, is_baseline "
            "FROM runs WHERE id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Run {run_id} not found")

        # Summary scores.
        score_cur = self._conn.cursor()
        score_cur.execute(
            "SELECT metric_name, mean_score, min_score, max_score, std_score "
            "FROM run_scores WHERE run_id = ?",
            (run_id,),
        )
        summary_scores = [dict(r) for r in score_cur.fetchall()]

        # Per-query details.
        detail_cur = self._conn.cursor()
        detail_cur.execute(
            "SELECT test_index, query, metric_name, score, reason "
            "FROM run_details WHERE run_id = ? ORDER BY test_index, metric_name",
            (run_id,),
        )
        details = [dict(r) for r in detail_cur.fetchall()]

        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "config": json.loads(row["config_json"]),
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "is_baseline": bool(row["is_baseline"]),
            "summary_scores": summary_scores,
            "details": details,
        }

    def compare_runs(self, run_id_1: str, run_id_2: str) -> dict:
        """Side-by-side comparison of two runs.

        Returns
        -------
        dict
            Keys: ``run_1``, ``run_2`` (full run dicts), and
            ``metric_comparison`` (list of dicts with metric_name,
            score_1, score_2, delta).
        """
        r1 = self.get_run_detail(run_id_1)
        r2 = self.get_run_detail(run_id_2)

        scores_1 = {s["metric_name"]: s["mean_score"] for s in r1["summary_scores"]}
        scores_2 = {s["metric_name"]: s["mean_score"] for s in r2["summary_scores"]}

        all_metrics = sorted(set(scores_1) | set(scores_2))
        comparison = []
        for metric in all_metrics:
            s1 = scores_1.get(metric, 0.0)
            s2 = scores_2.get(metric, 0.0)
            comparison.append(
                {
                    "metric_name": metric,
                    "score_1": s1,
                    "score_2": s2,
                    "delta": s2 - s1,
                }
            )

        return {
            "run_1": r1,
            "run_2": r2,
            "metric_comparison": comparison,
        }

    def detect_regression(
        self,
        current_scores: dict[str, float],
        threshold: float = 0.05,
    ) -> dict:
        """Compare current scores against the stored baseline.

        Parameters
        ----------
        current_scores : dict[str, float]
            Mapping of metric_name to the current mean score.
        threshold : float
            Maximum acceptable drop before flagging a regression.

        Returns
        -------
        dict
            Keys: ``regressions`` (list), ``improvements`` (list),
            ``stable`` (list), ``overall_passed`` (bool),
            ``baseline_id`` (str or None).
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id FROM runs WHERE is_baseline = 1 LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            return {
                "regressions": [],
                "improvements": [],
                "stable": list(current_scores.keys()),
                "overall_passed": True,
                "baseline_id": None,
            }

        baseline_id = row["id"]
        score_cur = self._conn.cursor()
        score_cur.execute(
            "SELECT metric_name, mean_score FROM run_scores WHERE run_id = ?",
            (baseline_id,),
        )
        baseline_scores = {r["metric_name"]: r["mean_score"] for r in score_cur.fetchall()}

        regressions, improvements, stable = [], [], []
        for metric, score in current_scores.items():
            baseline = baseline_scores.get(metric)
            if baseline is None:
                stable.append({"metric": metric, "current": score})
                continue
            delta = score - baseline
            entry = {"metric": metric, "current": score, "baseline": baseline, "delta": delta}
            if delta < -threshold:
                regressions.append(entry)
            elif delta > threshold:
                improvements.append(entry)
            else:
                stable.append(entry)

        return {
            "regressions": regressions,
            "improvements": improvements,
            "stable": stable,
            "overall_passed": len(regressions) == 0,
            "baseline_id": baseline_id,
        }

    def get_trend(self, metric_name: str, last_n: int = 10) -> list[dict]:
        """Return a time series of a metric across recent runs.

        Parameters
        ----------
        metric_name : str
            The metric to track.
        last_n : int
            Number of most recent runs to include.

        Returns
        -------
        list[dict]
            Each dict: ``run_id``, ``timestamp``, ``mean_score``.
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT r.id AS run_id, r.timestamp, rs.mean_score
            FROM runs r
            JOIN run_scores rs ON r.id = rs.run_id
            WHERE rs.metric_name = ?
            ORDER BY r.timestamp DESC
            LIMIT ?
            """,
            (metric_name, last_n),
        )
        rows = [dict(r) for r in cur.fetchall()]
        rows.reverse()  # Oldest first for charting.
        return rows

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
