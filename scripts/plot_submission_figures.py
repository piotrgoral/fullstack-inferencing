"""Focused Part B figures for submission.md.

    python scripts/plot_submission_figures.py \
      --input-csv results/experiments_with_completion_stats.csv \
      --output-dir figures/submission

Produces five figures, each tied to an insight in submission.md:
  F1  throughput by arm (per-user tok/s + system gen tok/s)
  F2  spec-decode acceptance vs realized e2e-p50 speedup (acceptance != speedup)
  F3  KV-cache saturation & queueing vs concurrency (baseline vs spec arms)
  F4  system-throughput / e2e-p95 tradeoff across concurrency
  F5  cost per completion token by arm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

# Ordering / styling reused from scripts/plot_experiment_metrics.py
MODE_ORDER = [
    "base-conv",
    "prefix-cache",
    "ngram-2",
    "dflash-2",
    "draft-q3-06b",
    "eagle3-2",
    "mtp-2",
]
MODEL_ORDER = ["q3_4b", "q35_4b"]
MODEL_LABEL = {"q3_4b": "Qwen3-4B", "q35_4b": "Qwen3.5-4B"}
SPEC_MODES = {"ngram-2", "dflash-2", "draft-q3-06b", "eagle3-2", "mtp-2"}
MODE_COLOR = {
    "base-conv": "#4C72B0",
    "prefix-cache": "#55A868",
    "ngram-2": "#C44E52",
    "dflash-2": "#8172B3",
    "draft-q3-06b": "#937860",
    "eagle3-2": "#DA8BC3",
    "mtp-2": "#CCB974",
}


def _mode_rank(mode: str) -> int:
    return MODE_ORDER.index(mode) if mode in MODE_ORDER else len(MODE_ORDER)


def _arms_for(df: pd.DataFrame, model: str) -> list[str]:
    present = set(df[df.model == model]["mode"].unique())
    return [m for m in MODE_ORDER if m in present]


# The committed CSV's cost column was computed at this GPU hourly rate.
CSV_COST_BASIS_HOURLY_USD = 2.5


def load(input_csv: Path, gpu_hourly_usd: float = CSV_COST_BASIS_HOURLY_USD) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    # Cost per token scales linearly with the GPU hourly rate, so rescale the
    # CSV's $2.50-basis cost to the requested rate (e.g. A10 @ $1.29/hr).
    scale = gpu_hourly_usd / CSV_COST_BASIS_HOURLY_USD
    df["cost_per_1M_usd"] = df["llm_gateway_cost_per_completion_token_usd"] * 1e6 * scale
    df.attrs["gpu_hourly_usd"] = gpu_hourly_usd
    return df


# ---------------------------------------------------------------- F1
def fig_throughput_by_arm(df: pd.DataFrame, out: Path) -> None:
    agg = df.groupby(["model", "mode"]).agg(
        user_tps=("llm_gateway_completion_tokens_per_second_p50", "mean"),
        sys_tps=("vllm_generation_tokens_total_rate_per_s", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, model in zip(axes, MODEL_ORDER):
        arms = _arms_for(df, model)
        sub = agg.loc[model].reindex(arms)
        x = range(len(arms))
        w = 0.38
        ax.bar([i - w / 2 for i in x], sub["sys_tps"], w,
               label="system gen tok/s (vLLM)", color="#4C72B0")
        ax.set_ylabel("system generation tok/s")
        ax2 = ax.twinx()
        ax2.bar([i + w / 2 for i in x], sub["user_tps"], w,
                label="per-user tok/s (gateway p50)", color="#DD8452")
        ax2.set_ylabel("per-user completion tok/s (p50)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(arms, rotation=30, ha="right")
        ax.set_title(MODEL_LABEL[model])
        ax.axhline(0, color="black", lw=0.5)
        # combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    fig.suptitle("F1 — Throughput by engine arm (mean over load configs)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- F2
def fig_acceptance_vs_speedup(df: pd.DataFrame, out: Path) -> None:
    base = (df[df["mode"] == "base-conv"]
            .groupby("model")["vllm_e2e_request_latency_seconds_p50"].mean())
    rows = []
    for (model, mode), g in df.groupby(["model", "mode"]):
        if mode not in SPEC_MODES:
            continue
        acc = g["vllm_spec_decode_accepted_over_draft_rate"].mean()
        e2e = g["vllm_e2e_request_latency_seconds_p50"].mean()
        speedup = base[model] / e2e  # >1 = faster than baseline
        rows.append((model, mode, acc, speedup))
    pts = pd.DataFrame(rows, columns=["model", "mode", "acc", "speedup"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, r in pts.iterrows():
        marker = "o" if r["model"] == "q3_4b" else "s"
        ax.scatter(r["acc"], r["speedup"], s=140, marker=marker,
                   color=MODE_COLOR.get(r["mode"], "#333"), edgecolor="black", zorder=3)
        ax.annotate(f"{r['mode']}\n({MODEL_LABEL[r['model']]})",
                    (r["acc"], r["speedup"]), textcoords="offset points",
                    xytext=(8, 6), fontsize=8)
    ax.axhline(1.0, color="red", ls="--", lw=1, label="baseline e2e-p50 (no speedup)")
    ax.set_xlabel("spec-decode acceptance  (accepted / draft tokens)")
    ax.set_ylabel("e2e-p50 speedup vs same-model baseline  (>1 = faster)")
    ax.set_title("F2 — Acceptance does not predict speedup")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- F3
def fig_kv_saturation(df: pd.DataFrame, out: Path) -> None:
    df = df.copy()
    df["arm_kind"] = df["mode"].apply(lambda m: "spec-decode" if m in SPEC_MODES else m)
    grp = df.groupby(["arm_kind", "max_sessions"]).agg(
        kv_max=("vllm_kv_cache_usage_perc_max", "mean"),
        wait_avg=("vllm_num_requests_waiting_avg", "mean"),
    ).reset_index()
    sessions = sorted(df["max_sessions"].unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    styles = {"base-conv": ("#4C72B0", "o"), "prefix-cache": ("#55A868", "^"),
              "spec-decode": ("#C44E52", "s")}
    for kind, (color, mk) in styles.items():
        s = grp[grp.arm_kind == kind].set_index("max_sessions").reindex(sessions)
        a1.plot(sessions, s["kv_max"], marker=mk, color=color, label=kind)
        a2.plot(sessions, s["wait_avg"], marker=mk, color=color, label=kind)
    a1.axhline(1.0, color="red", ls="--", lw=1, label="KV full")
    a1.set_xlabel("max_sessions (concurrency)"); a1.set_ylabel("KV-cache usage max")
    a1.set_title("KV-cache max vs concurrency"); a1.set_xticks(sessions)
    a1.legend(fontsize=9); a1.grid(True, alpha=0.3)
    a2.set_xlabel("max_sessions (concurrency)"); a2.set_ylabel("requests waiting (avg)")
    a2.set_title("Queue depth vs concurrency"); a2.set_xticks(sessions)
    a2.legend(fontsize=9); a2.grid(True, alpha=0.3)
    fig.suptitle("F3 — Spec decoding raises KV pressure → earlier saturation/queueing", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- F4
def fig_throughput_latency_tradeoff(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["mode"] == "base-conv"]
    grp = sub.groupby(["model", "max_sessions"]).agg(
        sys_tps=("vllm_generation_tokens_total_rate_per_s", "mean"),
        e2e_p95=("vllm_e2e_request_latency_seconds_p95", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    for model in MODEL_ORDER:
        s = grp[grp.model == model].sort_values("max_sessions")
        ax.plot(s["sys_tps"], s["e2e_p95"], marker="o", label=MODEL_LABEL[model])
        for _, r in s.iterrows():
            ax.annotate(f"s={int(r['max_sessions'])}", (r["sys_tps"], r["e2e_p95"]),
                        textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("system throughput (gen tok/s)")
    ax.set_ylabel("e2e latency p95 (s)")
    ax.set_title("F4 — Throughput/latency tradeoff across concurrency (baseline)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- F5
def fig_cost_per_token(df: pd.DataFrame, out: Path) -> None:
    agg = df.groupby(["model", "mode"])["cost_per_1M_usd"].mean()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.38
    base_models = MODEL_ORDER
    all_arms = sorted(df["mode"].unique(), key=_mode_rank)
    x = range(len(all_arms))
    for i, model in enumerate(base_models):
        vals = [agg.get((model, m), float("nan")) for m in all_arms]
        ax.bar([xi + (i - 0.5) * width for xi in x], vals, width,
               label=MODEL_LABEL[model],
               hatch="//" if model == "q3_4b" else "")
    ax.set_xticks(list(x)); ax.set_xticklabels(all_arms, rotation=30, ha="right")
    rate = df.attrs.get("gpu_hourly_usd", CSV_COST_BASIS_HOURLY_USD)
    ax.set_ylabel("USD per 1M completion tokens")
    ax.set_title(f"F5 — Cost per completion token by arm (GPU @ ${rate:.2f}/hr)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", type=Path,
                    default=Path("results/experiments_with_completion_stats.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("figures/submission"))
    ap.add_argument("--gpu-hourly", type=float, default=1.29,
                    help="GPU hourly USD rate for cost figure (CSV basis is $2.50)")
    args = ap.parse_args()

    df = load(args.input_csv, args.gpu_hourly)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig_throughput_by_arm(df, args.output_dir / "f1_throughput_by_arm.png")
    fig_acceptance_vs_speedup(df, args.output_dir / "f2_acceptance_vs_speedup.png")
    fig_kv_saturation(df, args.output_dir / "f3_kv_saturation.png")
    fig_throughput_latency_tradeoff(df, args.output_dir / "f4_throughput_latency_tradeoff.png")
    fig_cost_per_token(df, args.output_dir / "f5_cost_per_token.png")
    print(f"Wrote 5 figures to {args.output_dir}/")


if __name__ == "__main__":
    main()
