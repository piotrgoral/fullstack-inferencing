from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def load_rows_by_technique(csv_path: Path, technique_column: str = "technique") -> tuple[List[str], Dict[str, Dict[str, object]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV file has no header: {csv_path}")

        fieldnames = list(reader.fieldnames)
        if technique_column not in fieldnames:
            raise RuntimeError(f"Column '{technique_column}' not found in {csv_path}")

        by_technique: Dict[str, Dict[str, object]] = {}
        for row in reader:
            technique = str(row.get(technique_column, "")).strip()
            if not technique:
                continue
            # Last row for a given technique wins; experiments_merged should have one row per technique.
            by_technique[technique] = row

    return fieldnames, by_technique


def write_joined_csv(
    output_csv: Path,
    experiments_header: List[str],
    stats_header: List[str],
    joined_rows: List[Dict[str, object]],
    technique_column: str = "technique",
) -> None:
    # Drop join key and metadata column(s) we don't want in the output.
    drop_columns = {
        technique_column,
        "source_file",
        "server_profiles",
        "number_of_conversations",
    }
    exp_cols = [c for c in experiments_header if c not in drop_columns]
    stats_cols = [c for c in stats_header if c != technique_column]
    fieldnames = exp_cols + [c for c in stats_cols if c not in exp_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(joined_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join data/experiments_merged.csv with completion token stats CSV by "
            "technique using an inner join, dropping rows without matches and "
            "omitting the 'technique' column from the output."
        )
    )
    parser.add_argument(
        "--experiments-csv",
        type=Path,
        default=Path("data/experiments_merged.csv"),
        help="Path to experiments_merged CSV. Default: data/experiments_merged.csv",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=Path("data/completion_token_stats_all_gateway_logs.csv"),
        help=(
            "Path to completion token stats CSV. "
            "Default: data/completion_token_stats_all_gateway_logs.csv"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/experiments_with_completion_stats.csv"),
        help="Where to write the joined CSV. Default: data/experiments_with_completion_stats.csv",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        exp_header, exp_by_technique = load_rows_by_technique(args.experiments_csv)
        stats_header, stats_by_technique = load_rows_by_technique(args.stats_csv)

        # Add total_execution_time column (end_iso - start_iso) in seconds.
        for row in exp_by_technique.values():
            start_iso = row.get("start_iso")
            end_iso = row.get("end_iso")
            total_execution_time: float | None
            try:
                if start_iso and end_iso:
                    start_dt = datetime.fromisoformat(str(start_iso))
                    end_dt = datetime.fromisoformat(str(end_iso))
                    total_execution_time = (end_dt - start_dt).total_seconds()
                else:
                    total_execution_time = None
            except Exception:
                total_execution_time = None
            if total_execution_time is not None:
                row["total_execution_time"] = total_execution_time

        joined_rows: List[Dict[str, object]] = []
        for technique, exp_row in exp_by_technique.items():
            stats_row = stats_by_technique.get(technique)
            if stats_row is None:
                continue
            # Merge dictionaries; stats columns can override if there is any overlap.
            joined_rows.append({**exp_row, **stats_row})

        if not joined_rows:
            raise RuntimeError(
                "No overlapping techniques found between "
                f"{args.experiments_csv} and {args.stats_csv}"
            )

        write_joined_csv(args.output_csv, exp_header, stats_header, joined_rows)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Joined {len(joined_rows)} row(s) into {args.output_csv} "
        f"from {len(exp_by_technique)} experiment technique(s) and "
        f"{len(stats_by_technique)} stats technique(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

