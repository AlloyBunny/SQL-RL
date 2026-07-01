"""Thin wrapper around the vendored BIRD-RL critic evaluator."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sql_rl.data.six_gym import as_list
from sql_rl.prompts.sql_repair import extract_solution_sql


@dataclass(frozen=True)
class RewardResult:
    score: float
    correctness: float
    evaluation: dict[str, Any]
    invalid_sql: bool
    timeout: bool
    assertion_failed: bool
    sql_length: int


def evaluate_prediction(
    instance: dict[str, Any],
    pred_sqls: str | list[str] | None = None,
    *,
    db_dir: str | Path | None = None,
    mode: str = "pred",
) -> dict[str, Any]:
    """Evaluate a prediction with BIRD-RL's SQLite test-case evaluator.

    The template database is copied into a temporary directory first. This keeps
    raw dataset files clean while preserving BIRD-RL's expected
    `{db_id}_template.sqlite` / `{db_id}_ephemeral_1.sqlite` layout.
    """
    if mode not in {"pred", "gold"}:
        raise ValueError(f"mode must be 'pred' or 'gold', got {mode!r}")

    prepared = dict(instance)
    if pred_sqls is not None:
        prepared["pred_sqls"] = as_list(pred_sqls)
    elif "pred_sqls" in prepared:
        prepared["pred_sqls"] = as_list(prepared["pred_sqls"])
    elif "response" in prepared:
        prepared["pred_sqls"] = [extract_solution_sql(str(prepared["response"]))]

    for field in ("issue_sql", "sol_sql", "preprocess_sql", "clean_up_sql"):
        prepared[field] = as_list(prepared.get(field))
    prepared.setdefault("test_cases", [])
    prepared.setdefault("conditions", {})

    db_id = str(prepared.get("db_id") or prepared.get("selected_database") or "")
    if not db_id:
        raise ValueError("instance must contain db_id or selected_database")
    prepared.setdefault("db_id", db_id)

    template_path = _resolve_template_db(prepared, db_dir)

    evaluator = _load_vendored_evaluator()
    with tempfile.TemporaryDirectory(prefix=f"sql_rl_{db_id}_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        tmp_template = tmpdir_path / f"{db_id}_template.sqlite"
        tmp_ephemeral = tmpdir_path / f"{db_id}_ephemeral_1.sqlite"
        shutil.copy2(template_path, tmp_template)
        shutil.copy2(template_path, tmp_ephemeral)

        old_ephemeral = os.environ.get("EPHEMERAL_DB_PATH")
        old_db_dir = os.environ.get("SQL_REWARD_DB_DIR")
        os.environ["EPHEMERAL_DB_PATH"] = str(tmp_ephemeral)
        os.environ["SQL_REWARD_DB_DIR"] = str(tmpdir_path)
        try:
            args = SimpleNamespace(mode=mode)
            logger = evaluator.NullLogger()
            return evaluator.evaluate_instance(prepared, args, logger)
        finally:
            _restore_env("EPHEMERAL_DB_PATH", old_ephemeral)
            _restore_env("SQL_REWARD_DB_DIR", old_db_dir)


def reward_prediction(
    instance: dict[str, Any],
    response_or_sql: str | list[str],
    *,
    db_dir: str | Path | None = None,
    lambda_len: float = 0.0,
    lambda_invalid: float = 0.0,
    lambda_timeout: float = 0.0,
    max_sql_chars: int = 2000,
) -> RewardResult:
    """Return correctness-only or cost-aware reward for a model response."""
    pred_sqls = (
        response_or_sql
        if isinstance(response_or_sql, list)
        else [extract_solution_sql(response_or_sql)]
    )
    evaluation = evaluate_prediction(instance, pred_sqls, db_dir=db_dir, mode="pred")

    total = int(evaluation.get("total_test_cases") or 0)
    passed = int(evaluation.get("passed_test_cases") or 0)
    if total > 0:
        correctness = passed / total
    else:
        correctness = 1.0 if evaluation.get("status") == "success" else 0.0

    invalid_sql = bool(evaluation.get("evaluation_phase_execution_error"))
    timeout = bool(evaluation.get("evaluation_phase_timeout_error"))
    assertion_failed = bool(evaluation.get("evaluation_phase_assertion_error"))
    sql_length = sum(len(sql) for sql in pred_sqls)
    length_penalty = min(sql_length / max(max_sql_chars, 1), 1.0)
    score = (
        correctness
        - lambda_len * length_penalty
        - lambda_invalid * float(invalid_sql)
        - lambda_timeout * float(timeout)
    )

    return RewardResult(
        score=score,
        correctness=correctness,
        evaluation=evaluation,
        invalid_sql=invalid_sql,
        timeout=timeout,
        assertion_failed=assertion_failed,
        sql_length=sql_length,
    )


def _resolve_template_db(
    instance: dict[str, Any],
    db_dir: str | Path | None,
) -> Path:
    db_id = str(instance.get("db_id") or instance.get("selected_database"))
    candidates: list[Path] = []
    if instance.get("db_path"):
        candidates.append(Path(str(instance["db_path"])))
    if db_dir is not None:
        root = Path(db_dir)
        candidates.extend(
            [
                root / db_id / f"{db_id}_template.sqlite",
                root / "database" / db_id / f"{db_id}_template.sqlite",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates) or "<none>"
    raise FileNotFoundError(f"Could not find template database for {db_id}: {searched}")


def _load_vendored_evaluator():
    vendor_dir = Path(__file__).resolve().parents[1] / "vendor" / "bird_critic_eval"
    vendor_str = str(vendor_dir)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
    import single_instance_eval  # type: ignore

    return single_instance_eval


def _restore_env(name: str, old_value: str | None) -> None:
    if old_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old_value

