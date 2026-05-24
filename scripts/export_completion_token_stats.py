"""
# from all logs files
python3 scripts/export_completion_token_stats.py \
  --log-path logs/gateway \
  --output-csv data/completion_token_stats_all_gateway_logs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List


STAT_COLUMNS = [
    "count",
    "min",
    "p25",
    "p50",
    "p75",
    "p95",
    "p99",
    "max",
    "mean",
]


def _iter_gateway_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _discover_log_files(log_path: Path) -> List[Path]:
    if log_path.is_file():
        return [log_path]
    if log_path.is_dir():
        files = sorted(log_path.glob("gateway_metrics_*.jsonl"))
        if not files:
            raise RuntimeError(f"No gateway_metrics_*.jsonl files found in {log_path}")
        return files
    raise RuntimeError(f"Log path does not exist: {log_path}")


def _parse_completion_tokens(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _percentile(sorted_values: List[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * percentile
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    if lower_index == upper_index:
        return lower_value

    weight = rank - lower_index
    return lower_value + (upper_value - lower_value) * weight


def load_completion_tokens_by_technique(log_path: Path) -> Dict[str, List[float]]:
    by_technique: Dict[str, List[float]] = {}

    for file_path in _discover_log_files(log_path):
        for row in _iter_gateway_rows(file_path):
            technique = str(row.get("technique", "")).strip()
            if not technique:
                continue

            completion_tokens = _parse_completion_tokens(row.get("completion_tokens"))
            if completion_tokens is None:
                continue

            by_technique.setdefault(technique, []).append(completion_tokens)

    if not by_technique:
        raise RuntimeError("No rows with both technique and completion_tokens were found.")

    return by_technique


def summarize_completion_tokens(values: List[float]) -> Dict[str, float | int]:
    sorted_values = sorted(values)
    total = sum(sorted_values)
    count = len(sorted_values)

    return {
        "count": count,
        "min": sorted_values[0],
        "p25": _percentile(sorted_values, 0.25),
        "p50": _percentile(sorted_values, 0.50),
        "p75": _percentile(sorted_values, 0.75),
        "p95": _percentile(sorted_values, 0.95),
        "p99": _percentile(sorted_values, 0.99),
        "max": sorted_values[-1],
        "mean": total / count,
    }


def write_summary_csv(output_csv: Path, by_technique: Dict[str, List[float]]) -> int:
    fieldnames = ["technique", *STAT_COLUMNS]
    rows: List[Dict[str, object]] = []

    for technique in sorted(by_technique):
        summary = summarize_completion_tokens(by_technique[technique])
        rows.append({"technique": technique, **summary})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize completion_tokens per technique from gateway_metrics JSONL logs "
            "into a CSV."
        )
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/gateway"),
        help=(
            "Path to a gateway_metrics_*.jsonl file or to a directory containing them. "
            "Default: logs/gateway"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/completion_token_stats_by_technique.csv"),
        help=(
            "Where to write the output CSV. "
            "Default: data/completion_token_stats_by_technique.csv"
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        by_technique = load_completion_tokens_by_technique(args.log_path)
        technique_count = write_summary_csv(args.output_csv, by_technique)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote completion token stats for {technique_count} technique(s) to {args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
