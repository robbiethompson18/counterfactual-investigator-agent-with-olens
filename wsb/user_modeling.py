"""Text-only baseline on WorkspaceBench's user_modeling family: Gemini 3.8 Flash answers each item's chat as the assistant
(no lens, no activations), and WorkspaceBench's own judge decides whether the reply names the user attribute by inference.
This is the "what does a plain model reading the transcript get" floor the O-lens numbers should be compared against.

    source .envrc.local && uv run python -m wsb.user_modeling --n 3 --limit 10        # 5 selfdescribe + 5 synthsys, bank order
    uv run python -m wsb.user_modeling --items um-sd-country-00,um-ss-continent-02    # pick ids
    uv run python -m wsb.user_modeling --n 3                                          # all 100 items (300 + 300 Gemini calls)

Run dir: data/runs/wsb_user_modeling/<date>_<slug>/ with responses.jsonl, judged.jsonl, summary.json, meta.json and an
expview run (each item = one response + its judge call); `runs.py finish` syncs it to HF at the end.
Scoring per item: pass@1 = mean over samples, pass@n = any sample; `inferred` = picked gold with basis
inferred_characterization (their headline), `correct` = picked gold, `regex` = an `intermediates` alias appears in the reply."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from openai import APIStatusError

from counterfactual_investigator_agent_with_olens import openrouter
from counterfactual_investigator_agent_with_olens.runs import finish, new_run
from wsb import judge

sys.path.insert(0, str(Path(__file__).parents[1] / "expview"))
from expview import Run, call

BANK = Path(__file__).parent / "banks" / "user_modeling.json"
RESPONDER = "google/gemini-3.8-flash"
EFFORT = "medium"
TEMPERATURE = 1.0
METRICS = ("inferred", "correct", "regex")


def pick_items(bank: list[dict], items: str | None, limit: int) -> list[dict]:
    if items:
        want = items.split(",")
        return [it for it in bank if it["name"] in want]
    if limit:  # half per subfamily, bank order
        per = {"selfdescribe": (limit + 1) // 2, "synthsys": limit // 2}
        out = []
        for it in bank:
            if per[it["subfamily"]] > 0:
                out.append(it)
                per[it["subfamily"]] -= 1
        return out
    return bank


async def respond(item: dict, sample: int) -> dict:
    r = await openrouter.chat(RESPONDER, item["messages"], effort=EFFORT, temperature=TEMPERATURE, max_tokens=8000)
    msg = r.choices[0].message
    return {
        "id": item["name"],
        "sample": sample,
        "response": msg.content or "",
        "reasoning": getattr(msg, "reasoning", None),
        "usage": {"input": r.usage.prompt_tokens, "output": r.usage.completion_tokens} if r.usage else None,
    }


def summarize(items: list[dict], judged: list[dict]) -> dict:
    def flags(row):
        gold = row["pick"] == "gold"
        return {"inferred": gold and row["basis"] == judge.INFERRED, "correct": gold, "regex": row["regex"]}

    per_item = {}
    for it in items:
        rows = [flags(r) for r in judged if r["id"] == it["name"]]
        per_item[it["name"]] = {m: {"pass1": sum(f[m] for f in rows) / len(rows), "passn": any(f[m] for f in rows)} for m in METRICS}

    def table(names):
        return {
            "n_items": len(names),
            **{f"{m}_{k}": sum(per_item[n][m][k] for n in names) / len(names) for m in METRICS for k in ("pass1", "passn")},
        }

    by_sub = {sf: [it["name"] for it in items if it["subfamily"] == sf] for sf in ("selfdescribe", "synthsys")}
    return {
        "overall": table(list(per_item)),
        "by_subfamily": {sf: table(names) for sf, names in by_sub.items() if names},
        "per_item": per_item,
        "tokens": {
            "responder": {k: sum(r["usage"][k] for r in judged if r["usage"]) for k in ("input", "output")},
            "judge": {k: sum(r["judge_usage"][k] for r in judged if r["judge_usage"]) for k in ("input", "output")},
        },
    }


def print_table(summary: dict, n: int):
    print(f"{'':14}{'n':>4}" + "".join(f"{m + ' p@1':>16}{m + f' p@{n}':>16}" for m in METRICS))
    for name, t in [*summary["by_subfamily"].items(), ("overall", summary["overall"])]:
        print(f"{name:14}{t['n_items']:>4}" + "".join(f"{t[m + '_pass1']:>16.2f}{t[m + '_passn']:>16.2f}" for m in METRICS))


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3, help="samples per item")
    ap.add_argument("--limit", type=int, default=0, help="first k items, split evenly between subfamilies (0 = all)")
    ap.add_argument("--items", help="comma-separated item names instead of --limit")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--slug", default="flash-text-only")
    args = ap.parse_args()

    bank_file = json.loads(BANK.read_text())
    bank = bank_file["items"]
    items = pick_items(bank, args.items, args.limit)
    options = judge.build_options(bank)
    run = new_run(
        "wsb_user_modeling",
        args.slug,
        description=f"WorkspaceBench user_modeling text-only baseline: {RESPONDER} answers {len(items)} items x{args.n}, WSB um-v2 judge",
        models=[
            {"name": RESPONDER, "served_by": "openrouter", "role": "responder", "effort": EFFORT, "temperature": TEMPERATURE},
            {"name": judge.JUDGE_MODEL, "served_by": "openrouter", "role": "judge", "effort": judge.JUDGE_EFFORT},
        ],
        params={
            "n": args.n,
            "items": [it["name"] for it in items],
            "judge_prompt_version": judge.PROMPT_VERSION,
            "option_seed": judge.SEED,
            "bank_source": bank_file["source"],
        },
    )
    print(run)
    slots = asyncio.Semaphore(args.concurrency)
    view = Run(run / "expview", title=f"WSB user_modeling: {RESPONDER} text-only", description=__doc__, label_order=list(METRICS))

    async def one(item: dict, sample: int) -> dict:
        opts, gold_position = options[item["name"]]
        async with slots:
            # OpenRouter reports its new-account 20 rpm cap as a 402 (the dead BYOK keys' error comes first), so retry those too
            for attempt in range(20):
                try:
                    row = await respond(item, sample)
                    row |= await judge.judge(item, row["response"], opts, gold_position)
                    break
                except APIStatusError as e:
                    if e.status_code not in (402, 429) or attempt == 19:
                        raise
                    await asyncio.sleep(30)
        row |= {"subfamily": item["subfamily"], "attr": item["attr"], "options": opts, "gold_position": gold_position}
        reply = {"role": "assistant", "content": row["response"], "reasoning": row["reasoning"]}
        view.add(
            id=f"{item['name']}/s{sample}",
            group=item["subfamily"],
            labels={
                "inferred": row["pick"] == "gold" and row["basis"] == judge.INFERRED,
                "correct": row["pick"] == "gold",
                "regex": row["regex"],
                "pick": row["pick"],
                "basis": row["basis"],
            },
            calls=[
                call("subject", RESPONDER, item["messages"], output=reply),
                call(
                    "judge (WSB um-v2)",
                    judge.JUDGE_MODEL,
                    row["judge_messages"],
                    output={"role": "assistant", "content": row["judge_output"]},
                    parsed={k: row[k] for k in ("choice", "basis", "evidence", "pick")},
                ),
            ],
            extra={"attr": item["attr"], "gold_position": gold_position, "options": opts},
        )
        print(f"{item['name']} s{sample}: {row['pick']} ({row['basis']}) regex={row['regex']}", file=sys.stderr)
        return row

    judged = await asyncio.gather(*(one(it, s) for it in items for s in range(args.n)))
    judged.sort(key=lambda r: (r["id"], r["sample"]))
    jsonl = lambda rows: "".join(json.dumps(r) + "\n" for r in rows)
    (run / "responses.jsonl").write_text(jsonl({k: r[k] for k in ("id", "sample", "response", "reasoning", "usage")} for r in judged))
    (run / "judged.jsonl").write_text(jsonl({k: v for k, v in r.items() if k not in ("response", "reasoning")} for r in judged))
    summary = summarize(items, judged)
    (run / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print_table(summary, args.n)
    print("tokens:", summary["tokens"])
    finish(run)


if __name__ == "__main__":
    asyncio.run(main())
