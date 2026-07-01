"""Utilities for BIRD-Critic / SIX-GYM SQLite records."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SQL_LIST_FIELDS = {
    "issue_sql",
    "sol_sql",
    "preprocess_sql",
    "clean_up_sql",
    "pred_sqls",
}


def as_list(value: Any) -> list[str]:
    """Normalize JSON fields that may be missing, scalar strings, or lists."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item) for item in value if item is not None and str(item) != ""]
    return [str(value)]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def db_template_path(data_dir: str | Path, db_id: str) -> Path:
    """Return the SIX-GYM template SQLite path for a database id."""
    root = Path(data_dir)
    candidates = [
        root / "database" / db_id / f"{db_id}_template.sqlite",
        root / db_id / f"{db_id}_template.sqlite",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def normalize_record(
    record: dict[str, Any],
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize a raw SIX-GYM/BIRD-Critic record for SQL-RL code."""
    normalized = dict(record)
    for field in SQL_LIST_FIELDS:
        if field in normalized:
            normalized[field] = as_list(normalized[field])
    for field in ("issue_sql", "sol_sql", "preprocess_sql", "clean_up_sql"):
        normalized.setdefault(field, [])
    normalized.setdefault("test_cases", [])
    normalized.setdefault("conditions", {})

    db_id = normalized.get("db_id") or normalized.get("selected_database")
    if data_dir is not None and db_id and "db_path" not in normalized:
        normalized["db_path"] = str(db_template_path(data_dir, str(db_id)))
    return normalized


def iter_six_gym_records(
    data_dir: str | Path,
    filename: str = "train.jsonl",
) -> Iterator[dict[str, Any]]:
    """Yield normalized records from a SIX-GYM local directory."""
    data_root = Path(data_dir)
    jsonl_path = data_root / filename
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield normalize_record(json.loads(line), data_root)


def split_records(
    records: Sequence[dict[str, Any]],
    train_size: int = 4000,
    val_size: int = 500,
    test_size: int = 500,
    *,
    shuffle: bool = False,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create the fixed 4000/500/500 split used by the project docs."""
    items = list(records)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    expected = train_size + val_size + test_size
    if len(items) < expected:
        raise ValueError(f"Need at least {expected} records, got {len(items)}")
    train_end = train_size
    val_end = train_end + val_size
    test_end = val_end + test_size
    return items[:train_end], items[train_end:val_end], items[val_end:test_end]

