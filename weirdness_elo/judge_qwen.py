"""Re-judge every pair in run/comparisons.jsonl with Qwen3.6-27B (vLLM, OpenAI API) -> run/comparisons_qwen.jsonl.

One pair per request, same presentation order as the Sonnet judgement, temperature 0. One order of every pair
first, then the mirrored copies. Resumable: rows are keyed by (batch, a, b).
  python judge_qwen.py [--limit N] [--thinking] [--out PATH] [--conc 16]"""

import argparse
import asyncio
import json
import re
from pathlib import Path

import elo
import httpx

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.6-27B-FP8"
HEADER = elo.JUDGE_HEADER.split("You must pick one")[0].replace("For each pair below", "For the pair below")
INSTR = """You must pick one of A or B, even when close.

Answer with only a JSON object, no other text:
  {"winner": "A" or "B", "margin": "slight" | "clear" | "strong", "reason": "<at most 15 words>"}"""


def prompt(x, y, by_id):
    return f"{HEADER}{INSTR}\n\n### A\n{elo.render(by_id[x])}\n\n### B\n{elo.render(by_id[y])}\n"


def parse(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.DOTALL)
    m = re.findall(r"\{[^{}]*\}", txt)
    for s in reversed(m):
        try:
            d = json.loads(s)
        except Exception:  # noqa: BLE001, S112
            continue
        if str(d.get("winner", "")).strip().upper() in ("A", "B"):
            return d
    return None


async def main(a):
    run = Path("run")
    by_id = {it["id"]: it for it in elo.load_items(run)}
    comps = elo.load_comps(run)
    seen, first, second = set(), [], []
    for c in comps:
        k = frozenset((c["a"], c["b"]))
        (second if k in seen else first).append(c)
        seen.add(k)
    todo = first + second
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {(r["batch"], r["a"], r["b"]) for r in map(json.loads, filter(str.strip, out.read_text().splitlines()))}
    todo = [c for c in todo if (c["batch"], c["a"], c["b"]) not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"todo {len(todo)} (done {len(done)})", flush=True)
    sem = asyncio.Semaphore(a.conc)
    fh = open(out, "a")  # noqa: SIM115, ASYNC230 - one append handle for the whole run, closed at the end
    n = {"ok": 0, "bad": 0}
    async with httpx.AsyncClient(timeout=600) as cl:

        async def one(c):
            body = {
                "model": MODEL,
                "temperature": 0,
                "max_tokens": a.max_tokens,
                "messages": [{"role": "user", "content": prompt(c["a"], c["b"], by_id)}],
                "chat_template_kwargs": {"enable_thinking": a.thinking},
            }
            async with sem:
                for t in range(5):
                    try:
                        r = await cl.post(URL, json=body)
                        r.raise_for_status()
                        break
                    except Exception:  # noqa: BLE001
                        await asyncio.sleep(2 ** (t + 1))
                else:
                    n["bad"] += 1
                    return
            msg = r.json()["choices"][0]["message"]
            d = parse(msg.get("content"))
            row = {
                "a": c["a"],
                "b": c["b"],
                "first": c["a"],
                "round": c["round"],
                "batch": c["batch"],
                "sonnet_winner": c["winner"],
                "raw": (msg.get("content") or "")[-400:],
            }
            if d:
                row.update(winner=c["a"] if d["winner"].strip().upper() == "A" else c["b"], margin=d.get("margin"), reason=d.get("reason"))
                n["ok"] += 1
            else:
                row["winner"] = None
                n["bad"] += 1
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if (n["ok"] + n["bad"]) % 500 == 0:
                print(n, flush=True)

        await asyncio.gather(*(one(c) for c in todo))
    print("final", n, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--out", default="run/comparisons_qwen.jsonl")
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=200)
    asyncio.run(main(ap.parse_args()))
