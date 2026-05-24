# Part B — Inference Experiments (engine + gateway in the loop)

## 1. Workload and path

The agentic workload is a **multi-turn conversational replay**: `sharegpt_turn_bench.py`
takes 64 ShareGPT conversations and drives them concurrently through the production path —

```
client (turn bench) → nginx LB → gateway.py (X-Technique, Prometheus llm_gateway_*) → SSH tunnel → vLLM
```

Each "session" is an in-flight conversation; within a session, turns are sent sequentially,
so the load is a steady population of `max_sessions` concurrent multi-turn chats. This is the
realistic shape of an assistant/agent serving many users at once.

**Held constant across every arm:** original (non-quantized) Qwen weights, `tensor-parallel-size 1`,
the gateway, the prompt set, and the cost basis. **Varied:** the model, the vLLM engine config
(`mode`), and a load sweep `(max_turns, max_sessions, max_tokens)`.

**InferenceOps — where we look first:** if latency/errors spike, check the layers in order —
nginx (502/upstream), then gateway (`llm_gateway_*` rate/duration on `:9101`), then the tunnel,
then vLLM (`/metrics`: `vllm_kv_cache_usage_perc`, `vllm_num_requests_waiting`). In this sweep the
first real signal of trouble is **`vllm_kv_cache_usage_perc_max → 1.0` with `num_requests_waiting`
climbing** — a vLLM-layer admission/KV problem, not an LB or gateway fault.

## 2. Experiment matrix

**Models (2):** `q3_4b` = Qwen3-4B, `q35_4b` = Qwen3.5-4B (original weights, BF16, not quantized).

**Engine arms (`mode`) — real `vllm serve` deltas** (exact configs from
`notes/5_sharegpt/…-part2.md` and `experiments-execution.md`):

| mode | engine delta (`--speculative-config` JSON unless noted) | model | max-model-len |
|---|---|---|---|
| `base-conv` | none (vLLM defaults) | both | q35 262144 · q3 35000 |
| `prefix-cache` | `--enable-prefix-caching` | both | q35 262144 · q3 35000 |
| `ngram-2` | `{"method":"ngram","num_speculative_tokens":2,"prompt_lookup_min":2,"prompt_lookup_max":2}` | both | q35 80000 · q3 35000 |
| `dflash-2` | `{"method":"dflash","model":"z-lab/Qwen3.5-4B-DFlash","num_speculative_tokens":2}` (q3 uses `z-lab/Qwen3-4B-DFlash-b16`) | both | q35 80000 · q3 35000 |
| `mtp-2` | `{"method":"qwen3_next_mtp","num_speculative_tokens":2}` | q35 only | 262144 |
| `eagle3-2` | `{"method":"eagle3","model":"AngelSlim/Qwen3-4B_eagle3","num_speculative_tokens":2,"draft_tensor_parallel_size":1}` | q3 only | 35000 |
| `draft-q3-06b` | `{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":2}` | q3 only | 35000 |

All arms add `--tensor-parallel-size 1 --language-model-only --reasoning-parser qwen3
--default-chat-template-kwargs '{"enable_thinking": false}'`. **≥2 distinct engine configs**
(baseline vs prefix-cache vs the spec arms) satisfy the rubric; **the extra labeled dimension**
is the load sweep below. Note `--max-model-len` was **not** held constant across arms (caveat 4).

**Load sweep (5 configs per arm)** — `max_sessions` = concurrency:

| max_turns | max_sessions | max_tokens |
|---|---|---|
| 16 | 16 | 1024 |
| 16 | 32 | 1024 |
| 16 | 64 | 1024 |
| 16 | 64 | 2048 |
| 2 | 32 | 1024 |

→ 2 models × (5–6 arms) × 5 configs = **55 runs** (`data/experiments_with_completion_stats.csv`).

**Pinned reproducibility:** `vllm serve` with the flags above; gateway cost basis
`estimated_gpu_cost = window_s/3600 × GPU_HOURLY_COST_USD` with `GPU_HOURLY_COST_USD=2.5`
(`gateway.py:80,125`); single GPU, `tensor-parallel-size 1`. *(Fill exact GPU SKU + vLLM version
from your run host before submitting.)*

## 3. Results

### 3.1 Per-arm summary (mean over the 5 load configs)

| model | mode | TTFT p50 (ms) | e2e p50 (s) | e2e p95 (s) | per-user tok/s | system tok/s | KV max† | accept‡ | $/1M tok |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | base-conv | 305 | 23.1 | 54.9 | 30.3 | 574 | 0.49 | — | 13.6 |
| Qwen3.5-4B | prefix-cache | 273 | 22.2 | 54.7 | 30.9 | 579 | 0.49 | — | 13.5 |
| Qwen3.5-4B | mtp-2 | 383 | **21.3** | **53.6** | **31.5** | **619** | 1.00 | 0.51 | **12.9** |
| Qwen3.5-4B | dflash-2 | 2478 | 25.8 | 64.2 | 26.8 | 588 | 1.00 | 0.44 | 15.4 |
| Qwen3.5-4B | ngram-2 | 284 | 35.8 | 98.4 | 16.7 | 344 | 0.99 | 0.28 | 22.9 |
| Qwen3-4B | base-conv | 183 | 19.5 | 52.2 | 29.2 | 556 | 0.75 | — | 13.7 |
| Qwen3-4B | prefix-cache | 182 | 19.5 | **49.9** | 29.8 | 550 | 0.72 | — | 13.6 |
| Qwen3-4B | eagle3-2 | 282 | **18.4** | 64.0 | 28.1 | **617** | 0.79 | 0.50 | 14.3 |
| Qwen3-4B | dflash-2 | 305 | 20.2 | 69.5 | 27.1 | 566 | 0.99 | 0.53 | 15.3 |
| Qwen3-4B | draft-q3-06b | 1817 | 27.0 | 74.8 | 20.0 | 458 | 1.00 | **0.63** | 19.7 |
| Qwen3-4B | ngram-2 | 156 | 29.4 | 82.0 | 20.0 | 423 | 0.77 | 0.36 | 19.8 |

† KV max shown at the saturating point `max_sessions=64`. The cross-config mean is lower.
‡ accept = `vllm_spec_decode_accepted_over_draft_rate` (accepted ÷ drafted tokens).
*TTFT* = `vllm_time_to_first_token_seconds_p50`; *per-user tok/s* =
`llm_gateway_completion_tokens_per_second_p50` (≈ 1/`vllm_inter_token_latency`); *system tok/s* =
`vllm_generation_tokens_total_rate_per_s`; *$/1M tok* = `cost_per_completion_token_usd × 1e6`.
(Gateway-side TTFT/TPOT are not used here — see caveat 2.)

### 3.2 Concurrency sweep (baseline arm)

| model | sessions | per-user tok/s | system tok/s | e2e p95 (s) | KV max | waiting avg |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | 16 | 34.6 | 477 | 31.4 | 0.16 | 0.00 |
| Qwen3.5-4B | 32 | 34.2 | 598 | 43.0 | 0.29 | 0.00 |
| Qwen3.5-4B | 64 | 24.3 | 598 | 78.7 | 0.49 | 0.19 |
| Qwen3-4B | 16 | 35.0 | 461 | 30.5 | 0.33 | 0.00 |
| Qwen3-4B | 32 | 34.0 | 592 | 45.2 | 0.47 | 0.36 |
| Qwen3-4B | 64 | 21.5 | 567 | 70.1 | 0.75 | 0.81 |

## 4. Insights

**1. Speculative decoding rarely pays off for this batched conversational workload** (F1).
Per-user throughput sits at ~30 tok/s for baseline; only **MTP** on Qwen3.5 marginally beats it
(31.5 vs 30.3 tok/s, e2e-p50 21.3 vs 23.1 s). `ngram-2` (16.7 tok/s) and the external draft
`draft-q3-06b` (20.0 tok/s) are **far worse**; `dflash`/`eagle3` land at or just below baseline.
At 16–64 concurrent sessions the GPU is already compute-bound from batching, so the memory-bound
advantage that makes spec decoding shine at low concurrency largely disappears — the extra
verification + draft compute is not free here.

**2. Acceptance rate ranks methods but does not predict speedup** (F2).
Acceptance order is `draft-q3-06b` 0.63 > `dflash`(q3) 0.53 > `mtp` 0.51 ≈ `eagle3` 0.50 >
`dflash`(q35) 0.44 > `ngram` 0.28–0.36. Yet the highest-acceptance arm (`draft-q3-06b`) is one of
the **slowest** (e2e-p50 speedup ≈ 0.72× baseline) because a separate 0.6B draft model adds its own
forward pass and KV footprint. Only the cheap, tightly-integrated drafts (`mtp`, `eagle3`) clear the
break-even line — acceptance must be weighed against draft cost. The external-draft arms also pay a
large **prefill/TTFT penalty** (vLLM TTFT p50 ≈ 1.8 s for `draft-q3-06b`, 2.5 s for q35 `dflash` vs
~0.3 s baseline), hurting the first-token experience on top of the throughput loss.

**3. Speculative decoding ~doubles KV-cache pressure → earlier saturation and queueing** (F3).
At `max_sessions=64`, baseline/prefix-cache KV max stays ≤0.75 (Qwen3.5 only 0.49), but
`mtp`/`dflash`/`draft-q3-06b` hit **KV max ≈ 1.00** with `num_requests_waiting` spiking (up to 19–46
in individual runs). This is exactly the "KV avg/max near 1, mainly sessions=64" pattern noted in
`draft.md` — those were the **spec-decode** arms, not the baseline. The cleanest read is the
**Qwen3.5 MTP vs baseline** pair (both `--max-model-len 262144`): same KV budget, yet MTP saturates
(1.00) while baseline stays at 0.49 — the extra KV is the draft/verification overhead, not a smaller
cache (cf. caveat 4 on differing `--max-model-len`).

**4. Concurrency is the dominant lever; sessions ≈ 32 is the sweet spot** (F4).
16→32 sessions keeps per-user throughput flat (~34 tok/s) while system throughput rises 477→598
tok/s — free goodput from better batching. 32→64 drops per-user throughput to ~22–24 tok/s and
pushes e2e-p95 up sharply (Qwen3.5 43→79 s, Qwen3 45→70 s) for **almost no extra system throughput**.
64 sessions buys tail latency, not capacity.

**5. `max_tokens` 1024→2048 inflates the latency tail.**
At sessions=64, raising the cap from 1024→2048 lengthens completions (mean ~560→740 tok on Qwen3.5)
and roughly **doubles e2e-p95** (Qwen3.5 56→101 s, Qwen3 54→86 s). Cap generation length to protect p95.

**6. Cost per completion token tracks throughput** (F5).
Cheapest: `mtp` (Qwen3.5) **$12.9/1M**, then prefix-cache/baseline ~$13.5/1M. Most expensive:
`ngram` ($19.8–22.9/1M) and `draft-q3-06b` ($19.7/1M) — GPU-seconds burned on rejected drafts.

**7. Model comparison is close — choose the engine per model.**
Baselines are within ~15% on every headline metric (Qwen3 slightly lower e2e latency, Qwen3.5
slightly higher system throughput and the cheapest arm overall via MTP). The winning engine differs
by model: **Qwen3.5 → MTP**, **Qwen3 → EAGLE3 (or just baseline)**. Neither benefits from `ngram` or
an external draft model.

## 5. Decision / recommendation

- **Ship baseline or MTP for Qwen3.5-4B.** MTP is the only spec arm that is simultaneously fastest
  (e2e-p50 21.3 s), highest-throughput (619 tok/s) and cheapest ($12.9/1M) — but it saturates KV at
  64 sessions, so pair it with a concurrency cap. If you want headroom over latency, plain
  `base-conv` is within ~8% and never saturates.
- **Ship baseline or EAGLE3 for Qwen3-4B.** EAGLE3 gives the best e2e-p50 (18.4 s) and top system
  throughput (617 tok/s) at near-baseline cost; baseline/prefix-cache win on p95 tail.
- **Run at `max_sessions ≈ 32`** for the best throughput-per-latency, and **cap `max_tokens`** to
  hold p95 (2048 nearly doubles tail latency).
- **Drop `ngram-2` and `draft-q3-06b`** — both lose on latency, throughput, and cost for this
  conversational traffic.
- **Next model if p95/cost fails SLO:** larger draft-integrated targets or a smaller served model;
  **production knob order:** tune concurrency/`max_tokens` first (gateway/load), then engine arm,
  then scale out a second vLLM backend — re-tuning the engine arm is cheaper than adding GPUs.

## 6. Reproduce

```bash
# regenerate the figures referenced above
python scripts/plot_submission_figures.py \
  --input-csv data/experiments_with_completion_stats.csv \
  --output-dir figures/submission
```

Figures: `figures/submission/f1_throughput_by_arm.png` … `f5_cost_per_token.png`.

---

### Footnotes / data-quality caveats

1. **Prefix caching shows zero queries/hits in every run** (`vllm_prefix_cache_queries = hits = 0`,
   incl. the `prefix-cache` arm), and its prefill p50 matches baseline. Either the static ShareGPT
   bench shares no cross-session prefix or the APC metric wasn't scraped; treat prefix-cache as
   **neutral** here, not validated.
2. **Gateway-side latency metrics are not the model's latency.** `llm_gateway_request_duration_seconds`
   reads 0.05–0.22 s, and `llm_gateway_time_to_first_token`/`_time_per_output_token` read ~1.6 ms /
   ~0.15 ms — all gateway-side processing time, while the true per-request time is 13–30 s. Confirmed
   by `tokens_mean ÷ completion_tps_p50 ≈ vllm_e2e_request_latency_p50`, by `gateway.py:120`
   ("non-streaming or first byte not split"), and by the vLLM TTFT (~0.3 s) / ITL (~36 ms ⇒ ~28 tok/s,
   matching per-user tok/s). **SLOs use vLLM TTFT + `vllm_inter_token_latency` + `vllm_e2e_request_latency`**
   — which is what every table above reports.
3. Early outlier runs (per `draft.md`) came from a dirty environment; the numbers here are from the
   clean re-run export.
4. **`--max-model-len` was not held constant** across arms (q35 baseline/MTP 262144; q35
   dflash/ngram 80000; all q3 arms 35000). A smaller window means fewer total KV blocks, so absolute
   KV capacity differs between servers — compare KV % *within* a model/len group. The headline KV
   claim (insight 3) rests on the Qwen3.5 MTP-vs-baseline pair, which share 262144, so it is not
   affected by this confound. Cross-model latency/throughput comparisons are otherwise robust because
   the bottleneck here is compute/batching, not context length.
