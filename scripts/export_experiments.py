"""
python scripts/export_experiments.py \
  --log-path logs/gateway/gateway_metrics_2026-05-18.jsonl \
  --technique q35_4b_base-2 \
  --output-csv data/experiments_q35_4b_base-2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass
class TechniqueWindow:
    technique: str
    server_profiles: List[str]
    start: datetime
    end: datetime


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


def parse_technique_windows(
    log_path: str | Path,
    *,
    techniques: List[str] | None = None,
    padding_before_s: float = 5.0,
    padding_after_s: float = 10.0,
) -> Dict[str, TechniqueWindow]:
    """
    Scan gateway_metrics_*.jsonl rows and compute [t_start, t_end] windows per technique.
    """
    path = Path(log_path)
    files = _discover_log_files(path)

    # Normalize technique filter to lowercase for comparison but preserve original labels.
    requested: set[str] | None = None
    if techniques:
        requested = {t.strip().lower() for t in techniques if t.strip()}

    by_tech: Dict[str, Dict[str, Any]] = {}

    for fpath in files:
        for row in _iter_gateway_rows(fpath):
            tech = str(row.get("technique", "")).strip()
            if not tech:
                continue
            tech_key = tech.lower()
            if requested is not None and tech_key not in requested:
                continue

            ts_raw = row.get("timestamp")
            if not isinstance(ts_raw, str):
                continue
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                # Fallback for timestamps ending with "Z"
                try:
                    if ts_raw.endswith("Z"):
                        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    else:
                        continue
                except ValueError:
                    continue

            info = by_tech.setdefault(
                tech,
                {
                    "min_ts": ts,
                    "max_ts": ts,
                    "server_profiles": set(),
                },
            )
            if ts < info["min_ts"]:
                info["min_ts"] = ts
            if ts > info["max_ts"]:
                info["max_ts"] = ts

            sp = row.get("server_profile")
            if isinstance(sp, str) and sp:
                info["server_profiles"].add(sp)

    if not by_tech:
        raise RuntimeError("No matching techniques found in gateway metrics logs.")

    windows: Dict[str, TechniqueWindow] = {}
    for tech, info in by_tech.items():
        min_ts: datetime = info["min_ts"]
        max_ts: datetime = info["max_ts"]

        # Ensure timestamps are timezone-aware in UTC.
        if min_ts.tzinfo is None:
            min_ts = min_ts.replace(tzinfo=timezone.utc)
        if max_ts.tzinfo is None:
            max_ts = max_ts.replace(tzinfo=timezone.utc)

        start = min_ts - timedelta(seconds=padding_before_s)
        end = max_ts + timedelta(seconds=padding_after_s)
        windows[tech] = TechniqueWindow(
            technique=tech,
            server_profiles=sorted(info["server_profiles"]),
            start=start,
            end=end,
        )

    return windows


def _duration_to_seconds(spec: str) -> int:
    """
    Parse simple Prometheus-style duration strings like '30s', '5m', '1h' into seconds.
    """
    if not spec:
        raise ValueError("Empty duration spec")
    unit = spec[-1]
    try:
        value = float(spec[:-1])
    except ValueError as exc:
        raise ValueError(f"Invalid duration value: {spec}") from exc
    if unit == "s":
        return int(value)
    if unit == "m":
        return int(value * 60)
    if unit == "h":
        return int(value * 3600)
    raise ValueError(f"Unsupported duration unit in {spec!r} (use s, m, or h)")


def _prometheus_get(
    base_url: str,
    endpoint: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    query = urlencode(params)
    full_url = f"{url}?{query}"
    try:
        with urlopen(full_url) as resp:
            data = resp.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"Prometheus HTTP error {exc.code} for {full_url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Error connecting to Prometheus at {full_url}: {exc}") from exc

    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from Prometheus for {full_url}") from exc

    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus error: {payload}")
    return payload["data"]


def prom_query_range(
    base_url: str,
    query: str,
    start_ts: float,
    end_ts: float,
    step_s: int,
) -> List[Dict[str, Any]]:
    data = _prometheus_get(
        base_url,
        "/api/v1/query_range",
        {
            "query": query,
            "start": start_ts,
            "end": end_ts,
            "step": step_s,
        },
    )
    result = data.get("result") or []
    return result


def prom_query_instant(
    base_url: str,
    query: str,
    ts: float,
) -> List[Tuple[Dict[str, str], float]]:
    data = _prometheus_get(
        base_url,
        "/api/v1/query",
        {
            "query": query,
            "time": ts,
        },
    )
    result: List[Tuple[Dict[str, str], float]] = []
    for series in data.get("result") or []:
        metric = series.get("metric") or {}
        value = series.get("value")
        if not isinstance(value, list) or len(value) != 2:
            continue
        try:
            v = float(value[1])
        except (TypeError, ValueError):
            continue
        result.append((metric, v))
    return result


def _aggregate_time_series(series: List[Dict[str, Any]]) -> float | None:
    """
    Simple helper: flatten all values in a query_range result and return the average.
    """
    values: List[float] = []
    for s in series:
        for ts, v in s.get("values", []):
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                continue
    if not values:
        return None
    return sum(values) / len(values)


def export_experiments(
    *,
    log_path: str | Path,
    techniques: List[str] | None,
    prometheus_url: str,
    step: str,
    padding_before_s: float,
    padding_after_s: float,
    output_csv: str | Path,
) -> None:
    windows = parse_technique_windows(
        log_path,
        techniques=techniques,
        padding_before_s=padding_before_s,
        padding_after_s=padding_after_s,
    )
    rows: List[Dict[str, Any]] = []

    for tech, window in sorted(windows.items(), key=lambda kv: kv[0]):
        start_ts = window.start.timestamp()
        end_ts = window.end.timestamp()
        span_s = int(end_ts - start_ts)
        if span_s <= 0:
            span_s = _duration_to_seconds(step)

        label_selector = f'technique="{tech}"'
        window_spec = f"{span_s}s"

        # p50 / p95 / p99 gateway end-to-end latency (seconds) over the run
        # window using increase(histogram) / window_seconds.
        quantiles = {
            "llm_gateway_request_duration_seconds_p50": 0.50,
            "llm_gateway_request_duration_seconds_p95": 0.95,
            "llm_gateway_request_duration_seconds_p99": 0.99,
        }
        latency_metrics: Dict[str, float | None] = {}
        for key, q in quantiles.items():
            q_expr = (
                f"histogram_quantile({q}, "
                f"sum by (le) ("
                f"increase(llm_gateway_request_duration_seconds_bucket"
                f"{{{label_selector}}}[{window_spec}])))"
                f" / {span_s}"
            )
            vals = prom_query_instant(
                prometheus_url,
                q_expr,
                end_ts,
            )
            latency_metrics[key] = vals[0][1] if vals else None

        # Throughput (requests per second) over the run:
        throughput_expr = (
            "sum(increase(llm_gateway_requests_total"
            f"{{{label_selector}}}[{window_spec}])) / {span_s}"
        )
        throughput_vals = prom_query_instant(
            prometheus_url,
            throughput_expr,
            end_ts,
        )
        throughput_avg = throughput_vals[0][1] if throughput_vals else None

        # GPU cost over the window (USD) — same metric used in Grafana:
        # llm_gateway_estimated_gpu_cost_usd_total.
        cost_expr = (
            "sum(increase(llm_gateway_estimated_gpu_cost_usd_total"
            f"{{{label_selector}}}[{window_spec}]))"
        )
        cost_values = prom_query_instant(
            prometheus_url,
            cost_expr,
            end_ts,
        )
        gpu_cost_usd = cost_values[0][1] if cost_values else 0.0

        # Gateway TTFT and TPOT percentiles over the run window.
        ttft_quantiles = {
            "llm_gateway_time_to_first_token_seconds_p50": 0.50,
            "llm_gateway_time_to_first_token_seconds_p95": 0.95,
        }
        ttft_metrics: Dict[str, float | None] = {}
        for key, q in ttft_quantiles.items():
            ttft_expr = (
                f"histogram_quantile({q}, "
                f"sum by (le) ("
                f"increase(llm_gateway_time_to_first_token_seconds_bucket"
                f"{{{label_selector}}}[{window_spec}])))"
                f" / {span_s}"
            )
            vals = prom_query_instant(
                prometheus_url,
                ttft_expr,
                end_ts,
            )
            ttft_metrics[key] = vals[0][1] if vals else None

        tpot_quantiles = {
            "llm_gateway_time_per_output_token_seconds_p50": 0.50,
            "llm_gateway_time_per_output_token_seconds_p95": 0.95,
        }
        tpot_metrics: Dict[str, float | None] = {}
        for key, q in tpot_quantiles.items():
            tpot_expr = (
                f"histogram_quantile({q}, "
                f"sum by (le) ("
                f"increase(llm_gateway_time_per_output_token_seconds_bucket"
                f"{{{label_selector}}}[{window_spec}])))"
                f" / {span_s}"
            )
            vals = prom_query_instant(
                prometheus_url,
                tpot_expr,
                end_ts,
            )
            tpot_metrics[key] = vals[0][1] if vals else None

        # Streaming inter-chunk delay (SSE only).
        stream_inter_chunk_expr = (
            "histogram_quantile(0.95, "
            "sum by (le) ("
            "increase(llm_gateway_stream_inter_chunk_delay_seconds_bucket"
            f"{{{label_selector}}}[{window_spec}])))"
        )
        stream_inter_chunk_vals = prom_query_instant(
            prometheus_url,
            stream_inter_chunk_expr,
            end_ts,
        )
        llm_gateway_stream_inter_chunk_delay_seconds_p95 = (
            stream_inter_chunk_vals[0][1] if stream_inter_chunk_vals else None
        )

        stream_observations_expr = (
            "sum(increase(llm_gateway_stream_inter_chunk_delay_seconds_count"
            f"{{{label_selector}}}[{window_spec}])) / {span_s}"
        )
        stream_observations_vals = prom_query_instant(
            prometheus_url,
            stream_observations_expr,
            end_ts,
        )
        llm_gateway_stream_inter_chunk_delay_observations_per_s = (
            stream_observations_vals[0][1] if stream_observations_vals else None
        )

        # Prompt/completion token totals and rates over the run window.
        prompt_tokens_expr = (
            "sum(increase(llm_gateway_prompt_tokens_total"
            f"{{{label_selector}}}[{window_spec}]))"
        )
        prompt_tokens_vals = prom_query_instant(
            prometheus_url,
            prompt_tokens_expr,
            end_ts,
        )
        prompt_tokens_total = prompt_tokens_vals[0][1] if prompt_tokens_vals else 0.0

        completion_tokens_expr = (
            "sum(increase(llm_gateway_completion_tokens_total"
            f"{{{label_selector}}}[{window_spec}]))"
        )
        completion_tokens_vals = prom_query_instant(
            prometheus_url,
            completion_tokens_expr,
            end_ts,
        )
        completion_tokens_total = (
            completion_tokens_vals[0][1] if completion_tokens_vals else 0.0
        )

        prompt_tokens_rate = (
            prompt_tokens_total / span_s if span_s > 0 else 0.0
        )
        completion_tokens_rate = (
            completion_tokens_total / span_s if span_s > 0 else 0.0
        )

        # Completion tokens per second (proxy estimate from histogram).
        completion_tps_quantiles = {
            "llm_gateway_completion_tokens_per_second_p50": 0.50,
            "llm_gateway_completion_tokens_per_second_p95": 0.95,
        }
        completion_tps_metrics: Dict[str, float | None] = {}
        for key, q in completion_tps_quantiles.items():
            completion_tps_expr = (
                f"histogram_quantile({q}, "
                f"sum by (le) ("
                f"increase(llm_gateway_completion_tokens_per_second_bucket"
                f"{{{label_selector}}}[{window_spec}])))"
            )
            vals = prom_query_instant(
                prometheus_url,
                completion_tps_expr,
                end_ts,
            )
            completion_tps_metrics[key] = vals[0][1] if vals else None

        # Derived efficiency metrics (cost-per-token).
        total_tokens = prompt_tokens_total + completion_tokens_total
        cost_per_total_token = (
            gpu_cost_usd / total_tokens if total_tokens > 0 else None
        )
        cost_per_completion_token = (
            gpu_cost_usd / completion_tokens_total
            if completion_tokens_total > 0
            else None
        )

        # vLLM engine metrics (no technique label).
        kv_avg_expr = f"avg_over_time(vllm:kv_cache_usage_perc[{window_spec}])"
        kv_avg_values = prom_query_instant(
            prometheus_url,
            kv_avg_expr,
            end_ts,
        )
        avg_kv_cache_usage_perc = kv_avg_values[0][1] if kv_avg_values else 0.0

        kv_expr = f"max_over_time(vllm:kv_cache_usage_perc[{window_spec}])"
        kv_values = prom_query_instant(
            prometheus_url,
            kv_expr,
            end_ts,
        )
        max_kv_cache_usage_perc = kv_values[0][1] if kv_values else 0.0

        # Prefill and inter-token latency histograms over a short window (10s):
        # p50 ~ avg, p99 ~ max.
        prefill_avg_expr = (
            "histogram_quantile(0.50, "
            "sum by (le) (rate(vllm:request_prefill_time_seconds_bucket[5m])))"
        )
        prefill_max_expr = (
            "histogram_quantile(0.99, "
            "sum by (le) (rate(vllm:request_prefill_time_seconds_bucket[5m])))"
        )
        prefill_avg_vals = prom_query_instant(
            prometheus_url,
            prefill_avg_expr,
            end_ts,
        )
        prefill_max_vals = prom_query_instant(
            prometheus_url,
            prefill_max_expr,
            end_ts,
        )
        avg_vllm_request_prefill_time_s = (
            prefill_avg_vals[0][1] if prefill_avg_vals else 0.0
        )
        max_vllm_request_prefill_time_s = (
            prefill_max_vals[0][1] if prefill_max_vals else 0.0
        )

        inter_avg_expr = (
            "histogram_quantile(0.50, "
            "sum by (le) (rate(vllm:inter_token_latency_seconds_bucket[5m])))"
        )
        inter_max_expr = (
            "histogram_quantile(0.99, "
            "sum by (le) (rate(vllm:inter_token_latency_seconds_bucket[5m])))"
        )
        inter_avg_vals = prom_query_instant(
            prometheus_url,
            inter_avg_expr,
            end_ts,
        )
        inter_max_vals = prom_query_instant(
            prometheus_url,
            inter_max_expr,
            end_ts,
        )
        avg_vllm_inter_token_latency_s = (
            inter_avg_vals[0][1] if inter_avg_vals else 0.0
        )
        max_vllm_inter_token_latency_s = (
            inter_max_vals[0][1] if inter_max_vals else 0.0
        )

        # Time to first token latency histogram percentiles.
        tftt_p50_expr = (
            "histogram_quantile(0.50, "
            "sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))"
        )
        tftt_p95_expr = (
            "histogram_quantile(0.95, "
            "sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))"
        )
        tftt_p99_expr = (
            "histogram_quantile(0.99, "
            "sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))"
        )
        tftt_p50_vals = prom_query_instant(
            prometheus_url,
            tftt_p50_expr,
            end_ts,
        )
        tftt_p95_vals = prom_query_instant(
            prometheus_url,
            tftt_p95_expr,
            end_ts,
        )
        tftt_p99_vals = prom_query_instant(
            prometheus_url,
            tftt_p99_expr,
            end_ts,
        )
        vllm_time_to_first_token_seconds_p50 = (
            tftt_p50_vals[0][1] if tftt_p50_vals else 0.0
        )
        vllm_time_to_first_token_seconds_p95 = (
            tftt_p95_vals[0][1] if tftt_p95_vals else 0.0
        )
        vllm_time_to_first_token_seconds_p99 = (
            tftt_p99_vals[0][1] if tftt_p99_vals else 0.0
        )

        # Average generation throughput (tokens/s) during the run window.
        gen_total_expr = (
            f"sum(increase(vllm:generation_tokens_total[{window_spec}])) / {span_s}"
        )
        gen_vals = prom_query_instant(
            prometheus_url,
            gen_total_expr,
            end_ts,
        )
        total_vllm_generation_tokens = gen_vals[0][1] if gen_vals else 0.0

        row: Dict[str, Any] = {
            "technique": tech,
            "server_profiles": ",".join(window.server_profiles),
            "start_iso": window.start.isoformat(),
            "end_iso": window.end.isoformat(),
            "window_seconds": span_s,
            # Gateway (llm_gateway_*) metrics:
            "llm_gateway_request_duration_seconds_p50": latency_metrics.get(
                "llm_gateway_request_duration_seconds_p50"
            ),
            "llm_gateway_request_duration_seconds_p95": latency_metrics.get(
                "llm_gateway_request_duration_seconds_p95"
            ),
            "llm_gateway_request_duration_seconds_p99": latency_metrics.get(
                "llm_gateway_request_duration_seconds_p99"
            ),
            "llm_gateway_requests_total_rate_per_s": throughput_avg,
            "llm_gateway_estimated_gpu_cost_usd_total_increase": gpu_cost_usd,
            "llm_gateway_time_to_first_token_seconds_p50": ttft_metrics.get(
                "llm_gateway_time_to_first_token_seconds_p50"
            ),
            "llm_gateway_time_to_first_token_seconds_p95": ttft_metrics.get(
                "llm_gateway_time_to_first_token_seconds_p95"
            ),
            "llm_gateway_time_per_output_token_seconds_p50": tpot_metrics.get(
                "llm_gateway_time_per_output_token_seconds_p50"
            ),
            "llm_gateway_time_per_output_token_seconds_p95": tpot_metrics.get(
                "llm_gateway_time_per_output_token_seconds_p95"
            ),
            "llm_gateway_stream_inter_chunk_delay_seconds_p95": llm_gateway_stream_inter_chunk_delay_seconds_p95,
            "llm_gateway_stream_inter_chunk_delay_observations_per_s": llm_gateway_stream_inter_chunk_delay_observations_per_s,
            "llm_gateway_prompt_tokens_total_increase": prompt_tokens_total,
            "llm_gateway_completion_tokens_total_increase": completion_tokens_total,
            "llm_gateway_prompt_tokens_total_rate_per_s": prompt_tokens_rate,
            "llm_gateway_completion_tokens_total_rate_per_s": completion_tokens_rate,
            "llm_gateway_completion_tokens_per_second_p50": completion_tps_metrics.get(
                "llm_gateway_completion_tokens_per_second_p50"
            ),
            "llm_gateway_completion_tokens_per_second_p95": completion_tps_metrics.get(
                "llm_gateway_completion_tokens_per_second_p95"
            ),
            "llm_gateway_cost_per_total_token_usd": cost_per_total_token,
            "llm_gateway_cost_per_completion_token_usd": cost_per_completion_token,
            # vLLM engine metrics (vllm:*) aggregated over the window:
            "vllm_kv_cache_usage_perc_avg": avg_kv_cache_usage_perc,
            "vllm_kv_cache_usage_perc_max": max_kv_cache_usage_perc,
            "vllm_request_prefill_time_seconds_p50": avg_vllm_request_prefill_time_s,
            "vllm_request_prefill_time_seconds_p99": max_vllm_request_prefill_time_s,
            "vllm_inter_token_latency_seconds_p50": avg_vllm_inter_token_latency_s,
            "vllm_inter_token_latency_seconds_p99": max_vllm_inter_token_latency_s,
            "vllm_time_to_first_token_seconds_p50": vllm_time_to_first_token_seconds_p50,
            "vllm_time_to_first_token_seconds_p95": vllm_time_to_first_token_seconds_p95,
            "vllm_time_to_first_token_seconds_p99": vllm_time_to_first_token_seconds_p99,
            "vllm_generation_tokens_total_rate_per_s": total_vllm_generation_tokens,
        }
        row.update(latency_metrics)
        rows.append(row)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = [
        "technique",
        "server_profiles",
        "start_iso",
        "end_iso",
        "window_seconds",
        # Gateway (llm_gateway_*) metrics:
        "llm_gateway_requests_total_rate_per_s",
        "llm_gateway_estimated_gpu_cost_usd_total_increase",
        "llm_gateway_time_to_first_token_seconds_p50",
        "llm_gateway_time_to_first_token_seconds_p95",
        "llm_gateway_time_per_output_token_seconds_p50",
        "llm_gateway_time_per_output_token_seconds_p95",
        "llm_gateway_stream_inter_chunk_delay_seconds_p95",
        "llm_gateway_stream_inter_chunk_delay_observations_per_s",
        "llm_gateway_prompt_tokens_total_increase",
        "llm_gateway_completion_tokens_total_increase",
        "llm_gateway_prompt_tokens_total_rate_per_s",
        "llm_gateway_completion_tokens_total_rate_per_s",
        "llm_gateway_completion_tokens_per_second_p50",
        "llm_gateway_completion_tokens_per_second_p95",
        "llm_gateway_cost_per_total_token_usd",
        "llm_gateway_cost_per_completion_token_usd",
        "llm_gateway_request_duration_seconds_p50",
        "llm_gateway_request_duration_seconds_p95",
        "llm_gateway_request_duration_seconds_p99",
        # vLLM (vllm:*) metrics:
        "vllm_kv_cache_usage_perc_avg",
        "vllm_kv_cache_usage_perc_max",
        "vllm_request_prefill_time_seconds_p50",
        "vllm_request_prefill_time_seconds_p99",
        "vllm_inter_token_latency_seconds_p50",
        "vllm_inter_token_latency_seconds_p99",
        "vllm_time_to_first_token_seconds_p50",
        "vllm_time_to_first_token_seconds_p95",
        "vllm_time_to_first_token_seconds_p99",
        "vllm_generation_tokens_total_rate_per_s",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export experiment metrics per technique by combining gateway JSONL logs "
            "with Prometheus time-series (same metrics used in Grafana dashboards)."
        )
    )
    parser.add_argument(
        "--log-path",
        default="logs/gateway",
        help="Path to gateway_metrics_*.jsonl file or directory (default: logs/gateway).",
    )
    parser.add_argument(
        "--technique",
        action="append",
        dest="techniques",
        help=(
            "Technique label to export (can be passed multiple times). "
            "If omitted, all techniques seen in logs are used."
        ),
    )
    parser.add_argument(
        "--prometheus-url",
        default="http://127.0.0.1:9090",
        help="Base URL for Prometheus HTTP API (default: http://127.0.0.1:9090).",
    )
    parser.add_argument(
        "--step",
        default="30s",
        help="Step size for query_range (Prometheus duration, default: 30s).",
    )
    parser.add_argument(
        "--padding-before",
        type=float,
        default=5.0,
        help="Seconds before first log timestamp to include in Prometheus window.",
    )
    parser.add_argument(
        "--padding-after",
        type=float,
        default=10.0,
        help="Seconds after last log timestamp to include in Prometheus window.",
    )
    parser.add_argument(
        "--output-csv",
        default="experiments.csv",
        help="Path to output CSV file (default: experiments.csv).",
    )

    args = parser.parse_args(argv)

    try:
        export_experiments(
            log_path=args.log_path,
            techniques=args.techniques,
            prometheus_url=args.prometheus_url,
            step=args.step,
            padding_before_s=args.padding_before,
            padding_after_s=args.padding_after,
            output_csv=args.output_csv,
        )
        print(
            f"Wrote experiment metrics for techniques "
            f"{', '.join(sorted(parse_technique_windows(args.log_path, techniques=args.techniques).keys()))} "
            f"to {args.output_csv}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

