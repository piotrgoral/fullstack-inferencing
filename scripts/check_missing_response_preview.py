from __future__ import annotations

"""
Check gateway JSONL logs for entries that are missing the `response_preview`
field, either for a specific technique or for all techniques present.

Usage examples:

  python scripts/check_missing_response_preview.py \
    --log-path logs/gateway/gateway_metrics_2026-05-23.jsonl \
    --technique q3_4b_eagle3-2-num-conv-64-max-turns-16-max-sessions-16-max-tokens-1024

  python scripts/check_missing_response_preview.py \
    --log-path logs/gateway \
    --technique q35_4b_dflash-2-num-conv-64-max-turns-2-max-sessions-32-max-tokens-1024
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _iter_gateway_rows(log_path: Path) -> Iterable[Dict[str, Any]]:
    """
    Yield rows from a single gateway_metrics_*.jsonl file.
    """
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield row


def _discover_log_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("gateway_metrics_*.jsonl"))
        if not files:
            raise RuntimeError(f"No gateway_metrics_*.jsonl files found in {path}")
        return files
    raise RuntimeError(f"Log path does not exist: {path}")


def find_missing_response_previews(
    *,
    log_path: str | Path,
    technique: str | None,
) -> Dict[str, Tuple[int, List[Dict[str, Any]]]]:
    """
    Return, per technique, a tuple of (total_rows, sample_missing_rows).

    If `technique` is provided, only that technique is inspected.
    If `technique` is None, all techniques present in the logs are inspected.
    """
    path = Path(log_path)
    files = _discover_log_files(path)
    target = technique.strip().lower() if technique else None

    totals: Dict[str, int] = {}
    missing: Dict[str, List[Dict[str, Any]]] = {}

    for fpath in files:
        for row in _iter_gateway_rows(fpath):
            tech_raw = str(row.get("technique", "")).strip()
            if not tech_raw:
                continue
            tech_key = tech_raw.lower()

            if target is not None and tech_key != target:
                continue

            totals[tech_raw] = totals.get(tech_raw, 0) + 1

            # Treat rows as missing when `response_preview` key is absent or falsy.
            if not row.get("response_preview"):
                bucket = missing.setdefault(tech_raw, [])
                if len(bucket) < 20:
                    bucket.append(
                        {
                            "timestamp": row.get("timestamp"),
                            "status_code": row.get("status_code"),
                            "trace_id": row.get("trace_id"),
                            "conversation_id": row.get("conversation_id"),
                            "conversation_turn": row.get("conversation_turn"),
                        }
                    )

    result: Dict[str, Tuple[int, List[Dict[str, Any]]]] = {}
    for tech_label, total in totals.items():
        result[tech_label] = (total, missing.get(tech_label, []))
    return result


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Check gateway_metrics_*.jsonl logs for entries that are missing the "
        "`response_preview` field, either for a specific technique or all techniques."
    ))
    parser.add_argument(
        "--log-path",
        default="logs/gateway",
        help="Path to gateway_metrics_*.jsonl file or directory (default: logs/gateway).",
    )
    parser.add_argument(
        "--technique",
        help=(
            "Technique label to inspect (exact string as logged). "
            "If omitted, all techniques present in the logs are checked."
        ),
    )

    args = parser.parse_args(argv)

    try:
        by_tech = find_missing_response_previews(
            log_path=args.log_path,
            technique=args.technique,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}")
        return 1

    print(f"Log path: {args.log_path}")
    if not by_tech:
        print("No matching techniques found in logs.")
        return 0

    for tech, (total, missing) in sorted(by_tech.items(), key=lambda kv: kv[0]):
        print("\n" + "=" * 80)
        print(f"Technique: {tech}")
        print(f"Total rows for technique: {total}")
        print(f"Rows missing response_preview: {len(missing)}")

        if missing:
            print("First few missing rows (up to 20):")
            for i, row in enumerate(missing, start=1):
                ts = row.get("timestamp")
                status = row.get("status_code")
                trace_id = row.get("trace_id")
                conv_id = row.get("conversation_id")
                turn = row.get("conversation_turn")
                print(
                    f"{i:3d}. ts={ts} status={status} "
                    f"conversation_id={conv_id} turn={turn} trace_id={trace_id}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

