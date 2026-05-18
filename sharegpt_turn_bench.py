from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import Any

import httpx
from dotenv import load_dotenv

from sharegpt_dataset import load_sharegpt, iter_user_turns


def _resolve_gateway_base() -> str:
    """
    Resolve the OpenAI-compatible base URL for the gateway, mirroring crew.py.
    """
    served = os.environ.get("MODEL_NAME", "texttinyllama")
    base = os.environ.get("GATEWAY_OPENAI_BASE") or os.environ.get("OPENAI_API_BASE")
    if not base:
        use_lb = (
            os.environ.get("GATEWAY_USE_LOAD_BALANCER", "true").strip().lower()
            not in ("0", "false", "no", "off")
        )
        if use_lb:
            host = os.environ.get("GATEWAY_LB_HOST", "127.0.0.1")
            port = os.environ.get("GATEWAY_LB_PORT", "8780")
            base = f"http://{host}:{port}/v1"
        else:
            host = os.environ.get("GATEWAY_HOST", "127.0.0.1")
            port = os.environ.get("GATEWAY_PORT", "8765")
            base = f"http://{host}:{port}/v1"
    if not base.rstrip("/").endswith("/v1"):
        base = base.rstrip("/") + "/v1"

    # Print once so the user sees where traffic goes.
    print(f"Using gateway base {base!r} with model {served!r}")
    return base


async def _send_turn_request(
    client: httpx.AsyncClient,
    *,
    gateway_base: str,
    model: str,
    technique: str,
    conversation_id: int,
    turn_index: int,
    messages: list[dict[str, Any]],
    stream: bool,
    max_tokens: int,
    timeout_s: float,
) -> None:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens,
    }
    headers = {
        "X-Technique": technique,
        "X-Conversation-Id": str(conversation_id),
        "X-Conversation-Turn": str(turn_index),
        "Content-Type": "application/json",
    }

    url = gateway_base.rstrip("/") + "/chat/completions"
    started = time.perf_counter()
    try:
        if not stream:
            resp = await client.post(url, json=payload, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            elapsed = (time.perf_counter() - started) * 1000
            print(
                f"[conv {conversation_id} turn {turn_index}] "
                f"non-stream response HTTP {resp.status_code} in {elapsed:.1f} ms"
            )
            return

        async with client.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
            timeout=timeout_s,
        ) as response:
            response.raise_for_status()
            ttft_ms: float | None = None
            tokens = 0
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                now = time.perf_counter()
                if ttft_ms is None:
                    ttft_ms = (now - started) * 1000
                tokens += 1

        total_ms = (time.perf_counter() - started) * 1000
        note = (
            f"TTFT={ttft_ms:.1f} ms, total={total_ms:.1f} ms, "
            f"completion_tokens≈{tokens}"
            if ttft_ms is not None
            else f"total={total_ms:.1f} ms, completion_tokens≈{tokens}"
        )
        print(f"[conv {conversation_id} turn {turn_index}] {note}")
    except Exception as exc:
        total_ms = (time.perf_counter() - started) * 1000
        print(
            f"[conv {conversation_id} turn {turn_index}] "
            f"ERROR after {total_ms:.1f} ms: {exc}",
            file=sys.stderr,
        )


async def _run(args: argparse.Namespace) -> None:
    # Load dataset
    pool = load_sharegpt(
        path=args.dataset_path,
        min_input_tokens=args.min_input_tokens,
        max_input_tokens=args.max_input_tokens,
        num_conversations=args.num_conversations,
        seed=args.seed,
    )
    if not pool:
        print("No conversations loaded; exiting.")
        return

    gateway_base = _resolve_gateway_base()
    model = os.environ.get("MODEL_NAME", "texttinyllama")
    technique = args.technique

    timeout_s = float(args.timeout_s)

    async with httpx.AsyncClient() as client:
        for conv_id, item in enumerate(pool):
            if args.max_conversations is not None and conv_id >= args.max_conversations:
                break

            turns = iter_user_turns(
                item["messages"],
                mode=args.mode,
                max_turns=args.max_turns_per_conv,
            )
            if not turns:
                continue

            print(
                f"Conversation {conv_id}: {len(turns)} user turns "
                f"(estimated_input_tokens={item.get('estimated_input_tokens')})"
            )

            for t in turns:
                turn_index = int(t["turn_index"])
                messages = list(t["messages"])
                await _send_turn_request(
                    client,
                    gateway_base=gateway_base,
                    model=model,
                    technique=technique,
                    conversation_id=conv_id,
                    turn_index=turn_index,
                    messages=messages,
                    stream=args.stream,
                    max_tokens=args.max_tokens,
                    timeout_s=timeout_s,
                )
                if args.sleep_between_turns > 0:
                    await asyncio.sleep(args.sleep_between_turns)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay ShareGPT conversations turn-by-turn through the gateway "
            "to exercise metrics over multi-turn chat."
        )
    )
    parser.add_argument(
        "--technique",
        default="baseline",
        help="Technique label for X-Technique header (e.g. baseline, chunked_prefill).",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Path to ShareGPT JSON file (defaults to ~/.cache/llm_bench/sharegpt.json).",
    )
    parser.add_argument(
        "--min-input-tokens",
        type=int,
        default=100,
        help="Minimum estimated input tokens per conversation.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=2048,
        help="Maximum estimated input tokens per conversation.",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=100,
        help="Target number of conversations to sample from the dataset.",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Optional hard cap on conversations to replay (<= num-conversations).",
    )
    parser.add_argument(
        "--mode",
        choices=["static", "simulated"],
        default="static",
        help=(
            "Turn building mode: 'static' keeps dataset assistant turns in context, "
            "'simulated' only uses dataset user turns."
        ),
    )
    parser.add_argument(
        "--max-turns-per-conv",
        type=int,
        default=None,
        help="Optional cap on user turns per conversation.",
    )
    parser.add_argument(
        "--sleep-between-turns",
        type=float,
        default=0.2,
        help="Seconds to sleep between turns (helps make Grafana graphs legible).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Stream responses from the gateway (default: streaming enabled).",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming; use non-streaming chat completions.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="max_tokens parameter for each completion.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for dataset shuffling and sampling.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)


if __name__ == "__main__":
    main()

