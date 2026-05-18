from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def load_gateway_jsonl(log_dir: str | Path) -> pd.DataFrame:
    """
    Load gateway_metrics_*.jsonl files and return a DataFrame.

    Rows produced by sharegpt_turn_bench.py include conversation_id and
    conversation_turn so we can analyze per-turn metrics.
    """
    log_dir = Path(log_dir)
    rows: list[dict[str, Any]] = []
    for fpath in sorted(log_dir.glob("gateway_metrics_*.jsonl")):
        with fpath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No gateway_metrics_*.jsonl files found in {log_dir}")
    return pd.DataFrame(rows)


def plot_per_turn_latency(
    df: pd.DataFrame,
    *,
    technique: str = "baseline",
    server_profile: str | None = None,
) -> None:
    """
    Simple example: line plot of e2e latency per conversation turn.
    """
    mask = df["technique"] == technique
    if server_profile is not None:
        mask &= df["server_profile"] == server_profile
    df_f = df[mask].copy()
    if "conversation_id" not in df_f or "conversation_turn" not in df_f:
        raise RuntimeError("Dataframe missing conversation_id/conversation_turn columns.")

    df_f["conversation_turn"] = df_f["conversation_turn"].astype(int)
    df_f["conversation_id"] = df_f["conversation_id"].astype(int)

    df_f.sort_values(["conversation_id", "conversation_turn"], inplace=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    for conv_id, g in df_f.groupby("conversation_id"):
        ax.plot(
            g["conversation_turn"],
            g["e2e_latency_s"] * 1000.0,
            marker="o",
            label=f"conv {conv_id}",
            alpha=0.6,
        )
    ax.set_xlabel("Conversation turn (user index)")
    ax.set_ylabel("End-to-end latency (ms)")
    ax.set_title(f"E2E latency per turn — technique={technique}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df_all = load_gateway_jsonl("logs/gateway")
    plot_per_turn_latency(df_all, technique="baseline")

