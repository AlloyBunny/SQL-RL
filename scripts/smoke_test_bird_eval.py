#!/usr/bin/env python3
"""Run a local smoke test for the vendored BIRD critic evaluator."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sql_rl.rewards import reward_prediction


TEST_CASE = """
def test_case(pred_sqls, sol_sqls, db_path, conn, conditions):
    pred_res, pred_err, pred_timeout = execute_queries(pred_sqls, db_path, conn, None, "")
    sol_res, sol_err, sol_timeout = execute_queries(sol_sqls, db_path, conn, None, "")
    assert not pred_err and not pred_timeout
    assert not sol_err and not sol_timeout
    assert pred_res == sol_res, f"pred={pred_res}, sol={sol_res}"
    return 1
"""


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sql_rl_smoke_") as tmpdir:
        db_dir = Path(tmpdir) / "database" / "demo"
        db_dir.mkdir(parents=True)
        template_db = db_dir / "demo_template.sqlite"

        with sqlite3.connect(template_db) as conn:
            conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, salary INTEGER)")
            conn.executemany(
                "INSERT INTO employees (name, salary) VALUES (?, ?)",
                [("Ada", 120), ("Grace", 150), ("Linus", 90)],
            )
            conn.commit()

        instance = {
            "instance_id": "SMOKE_1",
            "db_id": "demo",
            "category": "Query",
            "query": "Return the name of the employee with the highest salary.",
            "issue_sql": ["SELECT name FROM employees ORDER BY salary ASC LIMIT 1"],
            "sol_sql": ["SELECT name FROM employees ORDER BY salary DESC LIMIT 1"],
            "preprocess_sql": [],
            "clean_up_sql": [],
            "test_cases": [TEST_CASE],
            "conditions": {"order": True},
            "db_path": str(template_db),
        }

        reward = reward_prediction(
            instance,
            "<solution>SELECT name FROM employees ORDER BY salary DESC LIMIT 1</solution>",
        )
        print(json.dumps(reward.evaluation, indent=2, sort_keys=True))
        print(f"score={reward.score:.3f} correctness={reward.correctness:.3f}")
        if reward.score != 1.0:
            raise SystemExit("smoke test failed")


if __name__ == "__main__":
    main()
