#!/usr/bin/env python3
"""Prepare fixed splits and prompt-ready JSONL files for SIX-GYM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sql_rl.data import iter_six_gym_records, split_records, write_jsonl
from sql_rl.env import extract_schema_text
from sql_rl.prompts import build_chat_messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = list(iter_six_gym_records(args.data_dir))
    train, val, test = split_records(
        records,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_cache: dict[tuple[str, tuple[str, ...]], str] = {}
    stats: dict[str, Any] = {
        "data_dir": args.data_dir,
        "output_dir": str(output_dir),
        "splits": {},
        "schema_errors": [],
    }

    for split_name, split_records_list in (
        ("train", train),
        ("val", val),
        ("test", test),
    ):
        prepared = [
            prepare_record(record, schema_cache, stats)
            for record in split_records_list
        ]
        write_jsonl(prepared, output_dir / f"{split_name}.jsonl")
        stats["splits"][split_name] = len(prepared)

    (output_dir / "prepare_report.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))


def prepare_record(
    record: dict[str, Any],
    schema_cache: dict[tuple[str, tuple[str, ...]], str],
    stats: dict[str, Any],
) -> dict[str, Any]:
    prepared = dict(record)
    db_path = str(prepared["db_path"])
    preprocess_sql = tuple(prepared.get("preprocess_sql") or [])
    cache_key = (db_path, preprocess_sql)
    if cache_key not in schema_cache:
        try:
            schema_cache[cache_key] = extract_schema_text(db_path, preprocess_sql)
        except Exception as exc:
            stats["schema_errors"].append(
                {
                    "instance_id": prepared.get("instance_id"),
                    "db_id": prepared.get("db_id"),
                    "error": str(exc),
                }
            )
            schema_cache[cache_key] = extract_schema_text(db_path, [])

    schema = schema_cache[cache_key]
    prepared["schema"] = schema
    prepared["messages"] = build_chat_messages(prepared, schema)
    prepared["target"] = "<solution>\n" + "\n\n".join(prepared.get("sol_sql") or []) + "\n</solution>"
    return prepared


if __name__ == "__main__":
    main()

