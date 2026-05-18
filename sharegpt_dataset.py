from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Any

SHAREGPT_URL = os.environ.get(
    "SHAREGPT_URL",
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
)

DEFAULT_CACHE_DIR = Path(
    os.environ.get("SHAREGPT_CACHE_DIR", Path.home() / ".cache" / "llm_bench")
)
DEFAULT_CACHE_PATH = DEFAULT_CACHE_DIR / "sharegpt.json"


def _download_sharegpt(url: str, path: Path) -> None:
    """
    Download the ShareGPT JSON from `url` into `path`.
    """
    print(f"Downloading ShareGPT dataset from {url} to {path} …")
    urllib.request.urlretrieve(url, path)
    print("Download complete.\n")


def _ensure_sharegpt(path: Path) -> Path:
    """
    Ensure the ShareGPT JSON exists at `path`, downloading it if needed.
    """
    path = path.expanduser()
    if path.is_file():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    _download_sharegpt(SHAREGPT_URL, path)
    return path


def _estimate_tokens_from_text(text: str) -> int:
    # Simple heuristic, consistent with vinayhpandya benchmark: words * 1.3
    return int(len(text.split()) * 1.3)


def load_sharegpt(
    path: Path | str | None = None,
    *,
    min_input_tokens: int = 100,
    max_input_tokens: int = 2048,
    num_conversations: int = 500,
    seed: int = 1234,
) -> list[dict[str, Any]]:
    """
    Load ShareGPT-style multi-turn conversations and return a pool of examples.

    Each item in the returned list has the shape:
        {
            "messages": [...],                  # full conversation as OpenAI messages
            "estimated_input_tokens": <int>,    # based on all user turns
        }
    """
    cache_path = Path(path) if path is not None else DEFAULT_CACHE_PATH
    dataset_path = _ensure_sharegpt(cache_path)

    with dataset_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    pool: list[dict[str, Any]] = []

    for entry in raw:
        conversations = entry.get("conversations", [])
        if not conversations:
            continue

        messages: list[dict[str, str]] = []
        for turn in conversations:
            role = turn.get("from", "")
            content = turn.get("value", "")
            if not content:
                continue
            if role == "human":
                messages.append({"role": "user", "content": content})
            elif role == "gpt":
                messages.append({"role": "assistant", "content": content})

        if not messages:
            continue

        # Estimate input tokens based on all user turns in the conversation.
        user_text = " ".join(m["content"] for m in messages if m.get("role") == "user")
        est = _estimate_tokens_from_text(user_text)
        if est < min_input_tokens or est > max_input_tokens:
            continue

        pool.append(
            {
                "id": entry.get("id"),
                "messages": messages,
                "estimated_input_tokens": est,
            }
        )

        if len(pool) >= num_conversations * 3:
            break

    if len(pool) == 0:
        print(
            "No ShareGPT conversations matched the filters; "
            "try relaxing min_input_tokens/max_input_tokens."
        )
        return []

    if len(pool) > num_conversations:
        pool = pool[:num_conversations]

    print(
        f"Loaded {len(pool)} ShareGPT conversations "
        f"(input tokens: {min_input_tokens}–{max_input_tokens})."
    )
    return pool


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the ShareGPT dataset JSON to a local path. "
            "By default this writes to ~/.cache/llm_bench/sharegpt.json."
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_CACHE_PATH),
        help="Output path for sharegpt.json "
        f"(default: {DEFAULT_CACHE_PATH}).",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=SHAREGPT_URL,
        help=(
            "URL to download the ShareGPT JSON from "
            "(default: value of SHAREGPT_URL or built-in default)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the output file already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """
    Simple CLI entrypoint so you can run:

        python sharegpt_dataset.py [--output PATH] [--url URL] [--force]
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    out_path = Path(args.output).expanduser()
    if out_path.is_file() and not args.force:
        print(f"ShareGPT dataset already exists at {out_path}. Use --force to re-download.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _download_sharegpt(args.url, out_path)


if __name__ == "__main__":
    main()


def iter_user_turns(
    messages: list[dict[str, Any]],
    *,
    mode: str = "static",
    max_turns: int | None = None,
) -> list[dict[str, Any]]:
    """
    Build a sequence of per-turn request payloads from a full conversation.

    Parameters
    ----------
    messages:
        Full ShareGPT conversation converted to OpenAI-style messages.
    mode:
        - \"static\": keep assistant turns from the dataset in context;
          each user turn yields a request that includes prior dataset
          user+assistant messages plus the current user turn.
        - \"simulated\": only user messages are taken from the dataset;
          the caller is expected to append the model's assistant reply
          between iterations if they want true simulated multi-turn flow.
    max_turns:
        Optional cap on the number of user turns to emit.

    Returns
    -------
    A list of dicts of the shape:
        {
            \"turn_index\": int,        # 0-based over user turns
            \"messages\": [...],        # messages to send for this turn
        }
    """
    mode = (mode or "static").lower()
    if mode not in {"static", "simulated"}:
        raise ValueError(f"Unsupported iter_user_turns mode: {mode!r}")

    results: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    user_turn_index = 0

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if not content or role not in {"user", "assistant"}:
            continue

        if role == "assistant":
            if mode == "static":
                history.append({"role": "assistant", "content": content})
            continue

        # role == "user"
        if mode == "static":
            current = history + [{"role": "user", "content": content}]
            history.append({"role": "user", "content": content})
        else:
            current = [{"role": "user", "content": content}]

        results.append(
            {
                "turn_index": user_turn_index,
                "messages": current,
            }
        )
        user_turn_index += 1

        if max_turns is not None and user_turn_index >= max_turns:
            break

    return results


