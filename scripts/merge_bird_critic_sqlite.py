#!/usr/bin/env python3
"""Merge BIRD-Critic-SQLite public data with emailed GT/test cases."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-dir", required=True)
    parser.add_argument("--email-zip", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    public_dir = Path(args.public_dir)
    public_jsonl = find_public_jsonl(public_dir)
    public_records = load_jsonl(public_jsonl)
    sol_records = read_sqlite_sol_from_email_zip(Path(args.email_zip))
    sol_by_id = {str(record["instance_id"]): record for record in sol_records}

    merged: list[dict[str, Any]] = []
    missing_solution: list[str] = []
    for record in public_records:
        instance_id = str(record.get("instance_id", ""))
        sol = sol_by_id.get(instance_id)
        if sol is None:
            missing_solution.append(instance_id)
            continue
        item = dict(record)
        item["sol_sql"] = sol.get("sol_sql", [])
        item["test_cases"] = sol.get("test_cases", [])
        normalize_db_fields(item, public_dir)
        merged.append(item)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in merged:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    extra_solutions = sorted(set(sol_by_id) - {str(r.get("instance_id", "")) for r in public_records})
    report = {
        "public_dir": str(public_dir),
        "public_jsonl": str(public_jsonl),
        "email_zip": args.email_zip,
        "public_rows": len(public_records),
        "solution_rows": len(sol_records),
        "merged_rows": len(merged),
        "missing_solution_count": len(missing_solution),
        "missing_solution_ids": missing_solution[:50],
        "extra_solution_count": len(extra_solutions),
        "extra_solution_ids": extra_solutions[:50],
        "output_jsonl": str(output_jsonl),
    }
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))

    if missing_solution or len(merged) != len(public_records):
        raise SystemExit(1)


def find_public_jsonl(public_dir: Path) -> Path:
    candidates = [
        path
        for path in public_dir.rglob("*.jsonl")
        if path.name != "sqlite_sol.jsonl" and "sol" not in path.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"No public JSONL found under {public_dir}")
    candidates.sort(key=lambda path: path.stat().st_size, reverse=True)
    return candidates[0]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_sqlite_sol_from_email_zip(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as outer:
        inner_bytes = outer.read("BIRD-Critic-sol/sqlite_sol.zip")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        raw = inner.read("sqlite_sol.jsonl").decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def normalize_db_fields(record: dict[str, Any], public_dir: Path) -> None:
    db_id = (
        record.get("db_id")
        or record.get("selected_database")
        or record.get("database")
    )
    if db_id:
        record.setdefault("db_id", db_id)
        template = find_template_db(public_dir, str(db_id))
        if template.exists():
            record["db_path"] = str(template)


def find_template_db(public_dir: Path, db_id: str) -> Path:
    candidates = [
        public_dir / "database" / db_id / f"{db_id}_template.sqlite",
        public_dir / db_id / f"{db_id}_template.sqlite",
        public_dir / "database" / db_id / f"{db_id}.sqlite",
        public_dir / db_id / f"{db_id}.sqlite",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


if __name__ == "__main__":
    main()

