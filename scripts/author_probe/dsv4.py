"""DeepSeek V4 Flash (OpenRouter, reasoning effort medium) answers the same author MCQs as Qwen, both option orders, for
the later Qwen-vs-DSv4 disagreement filter. Resumable: rows already in RUN/dsv4_answers.jsonl are skipped.

    uv run python -m scripts.author_probe.dsv4 RUN [--limit N] [--concurrency 32]
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

from counterfactual_investigator_agent_with_olens import openrouter
from scripts.author_probe.bank import BANK, LETTERS, answer_message, orders

MODEL = "deepseek/deepseek-v4-flash"


def parse(text: str, n: int) -> str | None:
    """A bare letter, else the last "Answer: X", else a standalone letter on the last line (DSv4 often explains first)."""
    t, ok = text.strip(), LETTERS[:n]
    m = re.fullmatch(rf"\W*([{ok}])\b\W*", t) or ([*re.finditer(rf"answer\W*([{ok}])\b", t, re.IGNORECASE)] or [None])[-1]
    last = t.splitlines()[-1] if t else ""
    m = m or re.match(rf"\W*([{ok}])[.):]", last) or re.search(rf"\b([{ok}])\b\W*$", last)
    return m.group(1) if m else None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.run) / "dsv4_answers.jsonl"
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    text_of = {p["id"]: p["text"] for p in map(json.loads, (BANK / "passages.jsonl").read_text().splitlines())}
    jobs = []
    for q in map(json.loads, (BANK / "questions.jsonl").read_text().splitlines()):
        for o, order in enumerate(orders(len(q["options"]))):
            if f"{q['id']}:o{o}" not in done:
                jobs.append((q, o, order))
    jobs = jobs[: a.limit]
    sem, cost, n_done = asyncio.Semaphore(a.concurrency), [0.0], [0]
    f = out.open("a")  # kept open for the whole run; rows are flushed one by one

    async def one(q, o, order):
        msg = answer_message(text_of[q["passage"]], q["question"], [q["options"][i] for i in order])
        async with sem:
            r = await openrouter.chat(MODEL, [{"role": "user", "content": msg}], effort="medium")
        text = r.choices[0].message.content or ""
        letter = parse(text, len(order))
        cost[0] += getattr(r.usage, "cost", 0) or 0
        row = {"id": f"{q['id']}:o{o}", "order": order, "content": text, "letter": letter}
        row |= {"option": order[LETTERS.index(letter)] if letter else None, "usage": r.usage.model_dump() if r.usage else None}
        f.write(json.dumps(row) + "\n")
        f.flush()
        n_done[0] += 1
        if n_done[0] % 500 == 0:
            print(f"{n_done[0]}/{len(jobs)} ${cost[0]:.2f}", flush=True)

    await asyncio.gather(*(one(*j) for j in jobs))
    print(f"done {n_done[0]}/{len(jobs)} ${cost[0]:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
