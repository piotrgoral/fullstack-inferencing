"""
python3 scripts/merge_experiment_csvs.py --output-csv data/experiments_merged.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List


FILENAME_RE = re.compile(
    r"^experiments_"
    r"(?P<model>.+?)_"
    r"(?P<mode>base-conv|dflash-2|dflash-8|draft-q3-06b-2|eagle3-2|mtp-2|ngram-2|prefix-cache)"
    r"-num-conv-(?P<number_of_conversations>\d+)"
    r"-max-turns-(?P<max_turns>\d+)"
    r"-max-sessions-(?P<max_sessions>\d+)"
    r"-max-tokens-(?P<max_tokens>\d+)$"
)

METADATA_COLUMNS = [
    "source_file",
    "model",
    "mode",
    "number_of_conversations",
    "max_turns",
    "max_sessions",
    "max_tokens",
]

MODE_NORMALIZATION = {
    "draft-q3-06b-2": "draft-q3-06b",
}


def parse_filename_metadata(csv_path: Path) -> Dict[str, str | int]:
    match = FILENAME_RE.match(csv_path.stem)
    if not match:
        raise ValueError(
            "Unsupported CSV filename format: "
            f"{csv_path.name}. Expected experiments_<model>_<mode>-num-conv-..."
        )

    number_of_conversations = int(match.group("number_of_conversations"))
    max_turns = int(match.group("max_turns"))
    max_sessions = int(match.group("max_sessions"))
    max_tokens = int(match.group("max_tokens"))
    mode = MODE_NORMALIZATION.get(match.group("mode"), match.group("mode"))

    metadata: Dict[str, str | int] = {
        "source_file": csv_path.name,
        "model": match.group("model"),
        "mode": mode,
        "number_of_conversations": number_of_conversations,
        "max_turns": max_turns,
        "max_sessions": max_sessions,
        "max_tokens": max_tokens,
    }
    return metadata


def iter_csv_paths(input_dir: Path, output_path: Path) -> List[Path]:
    csv_paths: List[Path] = []
    output_resolved = output_path.resolve()

    for csv_path in sorted(input_dir.glob("*.csv")):
        if csv_path.resolve() == output_resolved:
            continue
        if not FILENAME_RE.match(csv_path.stem):
            continue
        csv_paths.append(csv_path)

    if not csv_paths:
        raise RuntimeError(f"No matching experiment CSV files found in {input_dir}")
    return csv_paths


def load_rows(csv_paths: Iterable[Path]) -> tuple[List[str], List[Dict[str, object]]]:
    merged_rows: List[Dict[str, object]] = []
    discovered_columns: List[str] = []

    for csv_path in csv_paths:
        metadata = parse_filename_metadata(csv_path)

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise RuntimeError(f"CSV file has no header: {csv_path}")

            for fieldname in reader.fieldnames:
                if fieldname not in discovered_columns:
                    discovered_columns.append(fieldname)

            for row in reader:
                if not any(value not in ("", None) for value in row.values()):
                    continue
                merged_rows.append({**row, **metadata})

    return discovered_columns, merged_rows


def write_merged_csv(
    output_path: Path,
    source_columns: List[str],
    rows: List[Dict[str, object]],
) -> None:
    fieldnames = METADATA_COLUMNS + [col for col in source_columns if col not in METADATA_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(fieldnames=fieldnames, f=handle, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge experiment CSV files from data/ and append parsed metadata columns."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing experiment CSV files. Default: data",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/experiments_merged.csv"),
        help="Output CSV path. Default: data/experiments_merged.csv",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        csv_paths = iter_csv_paths(args.input_dir, args.output_csv)
        source_columns, rows = load_rows(csv_paths)
        write_merged_csv(args.output_csv, source_columns, rows)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Merged {len(rows)} rows from {len(csv_paths)} files into {args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
