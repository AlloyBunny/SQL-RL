#!/usr/bin/env python3
"""Run SQL evaluator sanity checks with existing SQL fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sql_rl.data.six_gym import as_list, db_template_path, load_jsonl, normalize_record
from sql_rl.rewards import reward_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--prediction-field", default="sol_sql")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument("--max-failures", type=int, default=None)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-results", default=None)
    args = parser.parse_args()

    records = load_jsonl(args.jsonl)
    selected = records[args.skip :]
    if args.limit is not None and args.limit >= 0:
        selected = selected[: args.limit]

    results: list[dict[str, Any]] = []
    for index, raw_record in enumerate(selected, start=args.skip):
        record = normalize_for_eval(raw_record, args.data_dir)
        prediction = as_list(record.get(args.prediction_field))
        item: dict[str, Any] = {
            "index": index,
            "instance_id": record.get("instance_id"),
            "db_id": record.get("db_id") or record.get("selected_database"),
        }
        if not prediction:
            item.update(
                {
                    "status": "failed",
                    "score": 0.0,
                    "correctness": 0.0,
                    "error": f"missing prediction field: {args.prediction_field}",
                }
            )
            results.append(item)
            continue

        try:
            reward = reward_prediction(record, prediction)
            item.update(
                {
                    "status": reward.evaluation.get("status"),
                    "score": reward.score,
                    "correctness": reward.correctness,
                    "invalid_sql": reward.invalid_sql,
                    "timeout": reward.timeout,
                    "assertion_failed": reward.assertion_failed,
                    "evaluation": reward.evaluation,
                }
            )
        except Exception as exc:
            item.update(
                {
                    "status": "exception",
                    "score": 0.0,
                    "correctness": 0.0,
                    "error": str(exc),
                }
            )
        results.append(item)

    summary = summarize(results, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))

    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_results:
        output_results = Path(args.output_results)
        output_results.parent.mkdir(parents=True, exist_ok=True)
        with output_results.open("w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")

    failed_count = summary["total"] - summary["successes"]
    below_rate = summary["success_rate"] < args.min_success_rate
    above_failures = (
        args.max_failures is not None and failed_count > args.max_failures
    )
    if below_rate or above_failures:
        raise SystemExit(1)


def normalize_for_eval(raw_record: dict[str, Any], data_dir: str) -> dict[str, Any]:
    record = normalize_record(raw_record, data_dir)
    db_id = str(record.get("db_id") or record.get("selected_database") or "")
    if db_id and "db_path" not in record:
        record["db_path"] = str(db_template_path(data_dir, db_id))
    return record


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    total = len(results)
    successes = sum(1 for item in results if item.get("status") == "success")
    exceptions = sum(1 for item in results if item.get("status") == "exception")
    invalid = sum(1 for item in results if item.get("invalid_sql"))
    timeouts = sum(1 for item in results if item.get("timeout"))
    assertion_failed = sum(1 for item in results if item.get("assertion_failed"))
    avg_score = (
        sum(float(item.get("score", 0.0)) for item in results) / total if total else 0.0
    )
    return {
        "jsonl": args.jsonl,
        "data_dir": args.data_dir,
        "prediction_field": args.prediction_field,
        "limit": args.limit,
        "skip": args.skip,
        "min_success_rate": args.min_success_rate,
        "max_failures": args.max_failures,
        "total": total,
        "successes": successes,
        "failures": total - successes,
        "success_rate": successes / total if total else 0.0,
        "avg_score": avg_score,
        "exceptions": exceptions,
        "invalid_sql": invalid,
        "timeouts": timeouts,
        "assertion_failed": assertion_failed,
        "failed_instance_ids": [
            item.get("instance_id") for item in results if item.get("status") != "success"
        ],
    }


if __name__ == "__main__":
    main()
