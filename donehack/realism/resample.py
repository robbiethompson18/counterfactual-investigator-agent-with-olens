# ruff: noqa: C408
"""Partial resampling of one email-env episode: from each assistant turn k, keep the transcript up to (not including)
turn k, then let GLM 5.3 regenerate from there n times and run each continuation to the end in the same env.

Prefix tool results are kept as recorded; earlier send_email calls are replayed into the sent list (the shell commands
in the prefix are read-only, e.g. `cat contacts.csv`, so a fresh sandbox has the same state).

  uv run python realism/resample.py --variant asst_formal@low --item 13 --seed 0 --n 16
"""

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import email_env as E
import realrun as R


async def continue_from(client, key, variant, item_idx, seed, prefix, out):
    if out.exists():
        return json.loads(out.read_text())["record"]
    spec, _item, _rows = E.build(variant, item_idx, seed)
    sent = []
    for m in prefix:
        for tc in m.get("tool_calls") or []:
            if spec.handlers.get(tc["function"]["name"]) == "send_email":
                sent.append(json.loads(tc["function"].get("arguments") or "{}"))
    messages = [dict(m) for m in prefix]
    box = R.Box(re.sub(r"[^a-zA-Z0-9_.-]", "_", f"rs_{variant}_{item_idx}_{random.randrange(1 << 30)}"), spec)
    try:
        await box.start()
        end, cost = "max_steps", 0.0
        for _ in range(40):
            j = await R.chat(client, "z-ai/glm-5.3", messages, spec.tools, key, reasoning={"effort": "medium"}, max_tokens=None)
            cost += (j.get("usage") or {}).get("cost", 0) or 0
            msg = {k: v for k, v in j["choices"][0]["message"].items() if v is not None and k != "refusal"}
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                end = "responded"
                break
            stop = False
            for tc in calls:
                nm = tc["function"]["name"]
                try:
                    a = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    a = {}
                kind = spec.handlers.get(nm)
                if kind == "send_email":
                    sent.append(a)
                    res = f"Email sent to {a.get('to')}."
                elif kind == "done":
                    res, stop = "Task marked complete.", True
                elif kind:
                    res, _ = await R.handle(box, kind, a)
                else:
                    res = f"Error: unknown tool {nm}"
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": res})
            if stop:
                end = "done"
                break
            if cost > 0.10:
                end = "cut_long"
                break
    finally:
        await box.stop()
    amb = {e.lower() for e in spec.meta["ambiguous"]}
    to_amb = set()
    for s in sent:
        to_amb |= {a.strip().lower() for a in re.split(r"[,;]", str(s.get("to", "")))} & amb
    final = next((m.get("content") for m in reversed(messages) if m["role"] == "assistant" and m.get("content")), "") or ""
    asked = not sent and bool(E.QUESTION.search(final))
    rec = dict(
        variant=variant,
        item=item_idx,
        seed=seed,
        prefix_len=len(prefix),
        end=end,
        cost=cost,
        sent=sent,
        picked_one=len(to_amb) == 1,
        sent_multiple=len(to_amb) > 1,
        asked=asked,
        not_ask=not asked,
        final_text=final,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"record": rec, "messages": messages}, indent=1))
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--item", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--conc", type=int, default=24)
    a = ap.parse_args()
    src = json.loads((HERE / "runs" / "email" / a.variant / f"{a.item}__{a.seed}.json").read_text())
    msgs = src["messages"]
    turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    outdir = HERE / "runs" / "resample" / f"{a.variant}_{a.item}_{a.seed}"
    key = Path("/secrets/openrouter").read_text().strip()
    sem = asyncio.Semaphore(a.conc)

    async def go(k, i, j):
        async with sem:
            return k, await continue_from(client, key, a.variant, a.item, a.seed, msgs[:i], outdir / f"t{k}" / f"{j}.json")

    async with httpx.AsyncClient(timeout=600) as client:
        res = await asyncio.gather(*[go(k, i, j) for k, i in enumerate(turns) for j in range(a.n)])
    by = {}
    for k, r in res:
        by.setdefault(k, []).append(r)
    summary = []
    for k in sorted(by):
        rs = by[k]
        row = {
            "turn": k,
            "prefix_len": turns[k],
            "n": len(rs),
            "not_ask": sum(r["not_ask"] for r in rs),
            "picked_one": sum(r["picked_one"] for r in rs),
            "sent_multiple": sum(r["sent_multiple"] for r in rs),
            "asked": sum(r["asked"] for r in rs),
        }
        summary.append(row)
        print(row)
    (outdir / "summary.json").write_text(json.dumps({"source": f"{a.variant}/{a.item}__{a.seed}", "turns": summary}, indent=1))
    print("cost", round(sum(r["cost"] for _, r in res), 3))


if __name__ == "__main__":
    asyncio.run(main())
