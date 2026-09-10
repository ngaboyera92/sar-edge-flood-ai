"""Disk-backed exact threshold-free metric store for frozen G4-07.

DuckDB is used so pooled/event average precision and operating-point diagnostics do
not require retaining billions of probabilities in RAM. Scores are stored as
FLOAT (float32), matching model-output precision, and targets are binary.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


class DuckDBScoreStore:
    def __init__(self, path):
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError("duckdb is required for disk-backed exact AP") from exc
        self.duckdb = duckdb
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path))
        self.con.execute("PRAGMA threads=1")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS scores(event_id VARCHAR, score FLOAT, target UTINYINT)"
        )

    def append(self, event_id, probability, target, valid=None):
        p = np.asarray(probability, dtype=np.float32).reshape(-1)
        y = np.asarray(target).reshape(-1)
        if p.shape != y.shape:
            raise ValueError("probability/target shape mismatch")
        if valid is None:
            v = np.ones(p.shape, dtype=bool)
        else:
            v = np.asarray(valid, dtype=bool).reshape(-1)
            if v.shape != p.shape:
                raise ValueError("valid shape mismatch")
        p = p[v]
        y = y[v]
        if p.size == 0:
            return 0
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise RuntimeError("Invalid flood probabilities")
        if not np.isin(y, (0, 1)).all():
            raise RuntimeError("Binary target must be 0/1")
        df = pd.DataFrame({
            "event_id": np.full(p.shape, str(event_id), dtype=object),
            "score": p,
            "target": y.astype(np.uint8, copy=False),
        })
        self.con.register("_batch", df)
        self.con.execute("INSERT INTO scores SELECT event_id, score, target FROM _batch")
        self.con.unregister("_batch")
        return int(p.size)

    def count(self):
        return int(self.con.execute("SELECT COUNT(*) FROM scores").fetchone()[0])

    def pooled_average_precision(self):
        row = self.con.execute(
            """
            WITH total AS (
                SELECT SUM(target)::DOUBLE AS p FROM scores
            ), grouped AS (
                SELECT score, COUNT(*)::DOUBLE AS n, SUM(target)::DOUBLE AS pos
                FROM scores GROUP BY score
            ), curve AS (
                SELECT score, n, pos,
                       SUM(pos) OVER (ORDER BY score DESC ROWS UNBOUNDED PRECEDING) AS tp,
                       SUM(n)   OVER (ORDER BY score DESC ROWS UNBOUNDED PRECEDING) AS rank
                FROM grouped
            )
            SELECT SUM((pos / total.p) * (tp / rank))
            FROM curve CROSS JOIN total
            WHERE total.p > 0
            """
        ).fetchone()
        if row is None or row[0] is None:
            raise RuntimeError("Pooled AP undefined with zero positive support")
        return float(row[0])

    def event_average_precision(self):
        rows = self.con.execute(
            """
            WITH totals AS (
                SELECT event_id, SUM(target)::DOUBLE AS p
                FROM scores GROUP BY event_id
            ), grouped AS (
                SELECT event_id, score, COUNT(*)::DOUBLE AS n, SUM(target)::DOUBLE AS pos
                FROM scores GROUP BY event_id, score
            ), curve AS (
                SELECT event_id, score, n, pos,
                       SUM(pos) OVER (PARTITION BY event_id ORDER BY score DESC ROWS UNBOUNDED PRECEDING) AS tp,
                       SUM(n)   OVER (PARTITION BY event_id ORDER BY score DESC ROWS UNBOUNDED PRECEDING) AS rank
                FROM grouped
            )
            SELECT curve.event_id,
                   SUM((curve.pos / totals.p) * (curve.tp / curve.rank)) AS ap
            FROM curve JOIN totals USING(event_id)
            WHERE totals.p > 0
            GROUP BY curve.event_id
            ORDER BY curve.event_id
            """
        ).fetchall()
        return {str(e): float(ap) for e, ap in rows}

    def operating_point_match(self, reference_precision, reference_recall):
        """Pooled A3 diagnostic. Exact distance ties resolve to higher threshold."""
        ref_p = float(reference_precision)
        ref_r = float(reference_recall)
        total_pos = self.con.execute("SELECT SUM(target)::DOUBLE FROM scores").fetchone()[0]
        if total_pos is None or total_pos <= 0:
            raise RuntimeError("Operating-point curve undefined with zero positive support")
        self.con.execute(
            """
            CREATE OR REPLACE TEMP VIEW _pr_curve AS
            WITH grouped AS (
                SELECT score, COUNT(*)::DOUBLE AS n, SUM(target)::DOUBLE AS pos
                FROM scores GROUP BY score
            )
            SELECT score AS threshold,
                   SUM(pos) OVER (ORDER BY score DESC ROWS UNBOUNDED PRECEDING)
                   / SUM(n) OVER (ORDER BY score DESC ROWS UNBOUNDED PRECEDING) AS precision,
                   SUM(pos) OVER (ORDER BY score DESC ROWS UNBOUNDED PRECEDING) / ? AS recall
            FROM grouped
            """,
            [float(total_pos)],
        )
        p_row = self.con.execute(
            "SELECT threshold, precision, recall FROM _pr_curve ORDER BY abs(precision-?) ASC, threshold DESC LIMIT 1",
            [ref_p],
        ).fetchone()
        r_row = self.con.execute(
            "SELECT threshold, precision, recall FROM _pr_curve ORDER BY abs(recall-?) ASC, threshold DESC LIMIT 1",
            [ref_r],
        ).fetchone()
        def pack(row):
            return {"threshold": float(row[0]), "precision": float(row[1]), "recall": float(row[2])}
        return {"precision_matched": pack(p_row), "recall_matched": pack(r_row)}

    def close(self):
        if self.con is not None:
            self.con.close()
            self.con = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
