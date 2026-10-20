"""One Claude Haiku 5.5 rollout per WeirdChat Qwen3.6-27B prompt (403), same messages minus Qwen's reply
(WeirdChat transcripts have no system prompt). Appends to data/claude.jsonl keyed by prompt_id; skips
prompts already done. The viewer shows them under each transcript.

    uv run --with pyarrow python weirdchat-viewer/scripts/claude_rollouts.py [concurrency]
"""

import asyncio
import glob
import json
import sys
from pathlib import Path

import anthropic
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/claude.jsonl"
MODEL = "claude-haiku-5-5"
CONCURRENCY = int(sys.argv[1]) if len(sys.argv) > 1 else 64


def prompts():
    """prompt_id -> the user turn(s) of its first transcript."""
    out = {}
    for f in sorted(glob.glob(str(ROOT / "data/hf/data/transcripts/qwen__qwen3.6-27b/*.parquet"))):
        for t in pq.read_table(f, columns=["prompt_id", "messages"]).to_pylist():
            out.setdefault(t["prompt_id"], [{"role": m["role"], "content": m["content"]} for m in t["messages"][:-1]])
    return out


async def main():
    done = {json.loads(line)["prompt_id"] for line in OUT.open()} if OUT.exists() else set()
    todo = {k: v for k, v in prompts().items() if k not in done}
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    client = anthropic.AsyncAnthropic(max_retries=8)
    sem = asyncio.Semaphore(CONCURRENCY)
    n_ok = 0

    async def one(pid, msgs):
        nonlocal n_ok
        async with sem:
            try:
                r = await client.messages.create(model=MODEL, max_tokens=16000, messages=msgs)
            except anthropic.APIError as e:
                print(f"{pid}: {type(e).__name__}: {e}", flush=True)
                return
        rec = {
            "prompt_id": pid,
            "model": MODEL,
            "text": "".join(b.text for b in r.content if b.type == "text"),
            "stop_reason": r.stop_reason,
            "stop_details": r.stop_details.to_dict() if r.stop_details else None,
            "usage": {"input": r.usage.input_tokens, "output": r.usage.output_tokens},
        }
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        n_ok += 1
        if n_ok % 50 == 0:
            print(f"{n_ok}/{len(todo)}", flush=True)

    await asyncio.gather(*(one(k, v) for k, v in todo.items()))
    print(f"wrote {n_ok}/{len(todo)} to {OUT}", flush=True)


asyncio.run(main())
