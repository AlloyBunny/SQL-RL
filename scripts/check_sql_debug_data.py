#!/usr/bin/env python3
"""Check SQL debugging dataset integrity."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = ("instance_id", "query", "issue_sql")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--expect-rows", type=int, default=None)
    parser.add_argument("--require-labels", action="store_true")
    parser.add_argument("--output-report", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    jsonl_path = Path(args.jsonl) if args.jsonl else data_dir / "train.jsonl"
    records = load_jsonl(jsonl_path)

    category_counts: Counter[str] = Counter()
    db_counts: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    missing_dbs: list[dict[str, str]] = []
    with_preprocess = 0
    with_cleanup = 0

    for record in records:
        for field in REQUIRED_FIELDS:
            if not record.get(field):
                missing_fields[field] += 1
        if args.require_labels:
            for field in ("sol_sql", "test_cases"):
                if not record.get(field):
                    missing_fields[field] += 1

        db_id = get_db_id(record)
        if db_id:
            db_counts[db_id] += 1
            db_path = find_template_db(data_dir, db_id)
            if not db_path.exists():
                missing_dbs.append(
                    {
                        "instance_id": str(record.get("instance_id", "")),
                        "db_id": db_id,
                        "expected": str(db_path),
                    }
                )
        else:
            missing_fields["db_id"] += 1

        category_counts[str(record.get("category", "unknown"))] += 1
        if record.get("preprocess_sql"):
            with_preprocess += 1
        if record.get("clean_up_sql"):
            with_cleanup += 1

    report: dict[str, Any] = {
        "data_dir": str(data_dir),
        "jsonl": str(jsonl_path),
        "rows": len(records),
        "expected_rows": args.expect_rows,
        "row_count_ok": args.expect_rows is None or len(records) == args.expect_rows,
        "categories": dict(sorted(category_counts.items())),
        "db_ids": dict(sorted(db_counts.items())),
        "num_db_ids": len(db_counts),
        "with_preprocess_sql": with_preprocess,
        "with_clean_up_sql": with_cleanup,
        "missing_fields": dict(sorted(missing_fields.items())),
        "missing_db_files": missing_dbs[:50],
        "missing_db_files_count": len(missing_dbs),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if args.output_report:
        output_path = Path(args.output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    has_errors = bool(missing_fields or missing_dbs or not report["row_count_ok"])
    if has_errors:
        raise SystemExit(1)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_db_id(record: dict[str, Any]) -> str:
    return str(
        record.get("db_id")
        or record.get("selected_database")
        or record.get("database")
        or ""
    )


def find_template_db(data_dir: Path, db_id: str) -> Path:
    candidates = [
        data_dir / "database" / db_id / f"{db_id}_template.sqlite",
        data_dir / db_id / f"{db_id}_template.sqlite",
        data_dir / "database" / db_id / f"{db_id}.sqlite",
        data_dir / db_id / f"{db_id}.sqlite",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


if __name__ == "__main__":
    main()

