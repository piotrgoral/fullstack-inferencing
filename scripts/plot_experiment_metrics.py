"""
python scripts/plot_experiment_metrics.py \
  --input-csv data/experiments_merged.csv \
  --token-stats-csv data/completion_token_stats_all_gateway_logs.csv \
  --output-dir data/experiments_charts
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


MODEL_COLUMN = "model"
MODE_COLUMN = "mode"
COMBO_COLUMNS = [
    "number_of_conversations",
    "max_turns",
    "max_sessions",
    "max_tokens",
]
METADATA_COLUMNS = [
    "source_file",
    MODEL_COLUMN,
    MODE_COLUMN,
    *COMBO_COLUMNS,
    "technique",
    "server_profiles",
    "start_iso",
    "end_iso",
    "window_seconds",
]
TOKENS_JOIN_KEY = "technique"
TOKENS_PREFIX = "tokens_"
TOTAL_EXECUTION_TIME_COLUMN = "total_execution_time"


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = float(stripped)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _mode_sort_key(mode: str) -> Tuple[int, str]:
    mode_priority = {
        "base-conv": 0,
        "prefix-cache": 1,
        "ngram-2": 2,
        "dflash-8": 3,
        "dflash-2": 4,
        "draft-q3-06b": 5,
        "eagle3-2": 98,
        "mtp-2": 99,
    }
    return (mode_priority.get(mode, 50), mode)


def _model_sort_key(model: str) -> Tuple[int, str]:
    model_priority = {
        "q3_4b": 0,
        "q35_4b": 1,
    }
    return (model_priority.get(model, 99), model)


def _model_hatch(model: str) -> str:
    if model == "q3_4b":
        return "//"
    return ""


def _combo_key(row: Dict[str, str]) -> Tuple[int, int, int, int]:
    return tuple(int(row[column]) for column in COMBO_COLUMNS)


def _combo_label(combo: Tuple[int, int, int, int]) -> str:
    conv, turns, sessions, tokens = combo
    return (
        f"conv={conv} | turns={turns} | sessions={sessions} | tokens={tokens}"
    )


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "metric"


def load_csv(input_csv: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV file has no header: {input_csv}")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def load_completion_token_stats(
    input_csv: Path,
) -> tuple[Dict[str, Dict[str, str]], List[str]]:
    rows, fieldnames = load_csv(input_csv)
    if TOKENS_JOIN_KEY not in fieldnames:
        raise RuntimeError(
            f"Token stats CSV must contain '{TOKENS_JOIN_KEY}': {input_csv}"
        )

    metric_columns = [name for name in fieldnames if name != TOKENS_JOIN_KEY]
    by_technique: Dict[str, Dict[str, str]] = {}
    for row in rows:
        technique = row.get(TOKENS_JOIN_KEY, "").strip()
        if not technique:
            continue
        by_technique[technique] = row
    return by_technique, metric_columns


def enrich_rows_with_token_stats(
    rows: List[Dict[str, str]],
    token_stats_by_technique: Dict[str, Dict[str, str]],
    token_metric_columns: List[str],
) -> None:
    for row in rows:
        technique = row.get(TOKENS_JOIN_KEY, "").strip()
        token_stats = token_stats_by_technique.get(technique)
        for metric_name in token_metric_columns:
            prefixed_name = f"{TOKENS_PREFIX}{metric_name}"
            row[prefixed_name] = token_stats.get(metric_name, "") if token_stats else ""


def enrich_rows_with_total_execution_time(rows: List[Dict[str, str]]) -> None:
    for row in rows:
        start_iso = row.get("start_iso", "").strip()
        end_iso = row.get("end_iso", "").strip()
        if not start_iso or not end_iso:
            row[TOTAL_EXECUTION_TIME_COLUMN] = ""
            continue
        try:
            start_dt = datetime.fromisoformat(start_iso)
            end_dt = datetime.fromisoformat(end_iso)
        except ValueError:
            row[TOTAL_EXECUTION_TIME_COLUMN] = ""
            continue
        row[TOTAL_EXECUTION_TIME_COLUMN] = str((end_dt - start_dt).total_seconds())


def discover_numeric_columns(
    rows: List[Dict[str, str]],
    fieldnames: List[str],
) -> List[str]:
    numeric_columns: List[str] = []
    for fieldname in fieldnames:
        if fieldname in METADATA_COLUMNS:
            continue
        parsed_values = [_parse_float(row.get(fieldname)) for row in rows]
        if any(value is not None for value in parsed_values):
            numeric_columns.append(fieldname)
    return numeric_columns


def build_metric_lookup(
    rows: List[Dict[str, str]],
    metric_name: str,
) -> Dict[Tuple[str, Tuple[int, int, int, int], str], float]:
    values: Dict[Tuple[str, Tuple[int, int, int, int], str], float] = {}
    for row in rows:
        mode = row.get(MODE_COLUMN, "").strip()
        model = row.get(MODEL_COLUMN, "").strip()
        if not mode or not model:
            continue
        value = _parse_float(row.get(metric_name))
        if value is None:
            continue
        values[(mode, _combo_key(row), model)] = value
    return values


def filter_rows_excluding_sessions(
    rows: List[Dict[str, str]],
    excluded_sessions: set[int],
) -> List[Dict[str, str]]:
    filtered_rows: List[Dict[str, str]] = []
    for row in rows:
        sessions_raw = row.get("max_sessions", "").strip()
        try:
            sessions = int(sessions_raw)
        except ValueError:
            continue
        if sessions in excluded_sessions:
            continue
        filtered_rows.append(row)
    return filtered_rows


def plot_metric(
    rows: List[Dict[str, str]],
    metric_name: str,
    output_dir: Path,
) -> Path | None:
    metric_values = build_metric_lookup(rows, metric_name)
    if not metric_values:
        return None

    modes = sorted({mode for mode, _combo, _model in metric_values}, key=_mode_sort_key)
    combos = sorted({combo for _mode, combo, _model in metric_values})
    models = sorted({model for _mode, _combo, model in metric_values}, key=_model_sort_key)
    if not modes or not combos or not models:
        return None

    color_map = plt.get_cmap("tab20", len(combos))
    figure_width = max(14, len(modes) * 2.8)
    figure_height = max(6, 4 + len(combos) * 0.22)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    bar_group_width = 0.8
    series: List[Tuple[Tuple[int, int, int, int], str]] = [
        (combo, model) for combo in combos for model in models
    ]
    bar_width = bar_group_width / max(len(series), 1)
    centers = list(range(len(modes)))

    for series_index, (combo, model) in enumerate(series):
        offsets = [
            center - (bar_group_width / 2) + (series_index + 0.5) * bar_width
            for center in centers
        ]
        heights = [
            metric_values.get((mode, combo, model), math.nan)
            for mode in modes
        ]
        ax.bar(
            offsets,
            heights,
            width=bar_width * 0.95,
            color=color_map(combos.index(combo)),
            hatch=_model_hatch(model),
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_title(metric_name)
    ax.set_xlabel("mode")
    ax.set_ylabel(metric_name)
    ax.set_xticks(centers)
    ax.set_xticklabels(modes, rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    combo_handles = [
        Patch(facecolor=color_map(index), edgecolor="black", label=_combo_label(combo))
        for index, combo in enumerate(combos)
    ]
    combo_legend = fig.legend(
        handles=combo_handles,
        title="Metadata combination",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        borderaxespad=0.0,
        fontsize="small",
        ncol=max(1, min(3, len(combo_handles))),
    )
    ax.add_artist(combo_legend)

    model_handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=_model_hatch(model),
            label=model,
        )
        for model in models
    ]
    ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.7),
        borderaxespad=0.0,
        fontsize="small",
    )

    fig.subplots_adjust(right=0.82, bottom=0.28)
    output_path = output_dir / f"{_sanitize_filename(metric_name)}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one grouped bar chart per numeric metric from "
            "data/experiments_merged.csv."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/experiments_merged.csv"),
        help="Merged experiment CSV to visualize. Default: data/experiments_merged.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments_charts"),
        help="Directory for generated chart PNG files. Default: data/experiments_charts",
    )
    parser.add_argument(
        "--token-stats-csv",
        type=Path,
        default=Path("data/completion_token_stats_all_gateway_logs.csv"),
        help=(
            "Optional token stats CSV to join by technique and plot as tokens_* metrics. "
            "Default: data/completion_token_stats_all_gateway_logs.csv"
        ),
    )
    parser.add_argument(
        "--small-batch-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for a second chart set that excludes rows with "
            "max_sessions 64. Default: sibling 'small_batch' directory "
            "next to --output-dir."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    small_batch_dir = (
        args.small_batch_dir
        if args.small_batch_dir is not None
        else args.output_dir.parent / "small_batch"
    )

    try:
        rows, fieldnames = load_csv(args.input_csv)
        if not rows:
            raise RuntimeError(f"No data rows found in {args.input_csv}")

        enrich_rows_with_total_execution_time(rows)
        if TOTAL_EXECUTION_TIME_COLUMN not in fieldnames:
            fieldnames.append(TOTAL_EXECUTION_TIME_COLUMN)

        if args.token_stats_csv.exists():
            token_stats_by_technique, token_metric_columns = load_completion_token_stats(
                args.token_stats_csv
            )
            enrich_rows_with_token_stats(
                rows,
                token_stats_by_technique,
                token_metric_columns,
            )
            fieldnames.extend(
                f"{TOKENS_PREFIX}{metric_name}"
                for metric_name in token_metric_columns
                if f"{TOKENS_PREFIX}{metric_name}" not in fieldnames
            )

        numeric_columns = discover_numeric_columns(rows, fieldnames)
        if not numeric_columns:
            raise RuntimeError("No numeric metric columns found to plot.")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        small_batch_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        for metric_name in numeric_columns:
            if plot_metric(rows, metric_name, args.output_dir) is not None:
                generated += 1

        small_batch_rows = filter_rows_excluding_sessions(rows, {64})
        small_batch_generated = 0
        for metric_name in numeric_columns:
            if plot_metric(small_batch_rows, metric_name, small_batch_dir) is not None:
                small_batch_generated += 1

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Generated {generated} chart(s) from {args.input_csv} into {args.output_dir}"
    )
    print(
        "Generated "
        f"{small_batch_generated} small_batch chart(s) from {args.input_csv} "
        f"into {small_batch_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
