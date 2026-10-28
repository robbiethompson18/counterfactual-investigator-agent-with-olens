"""Text-only baseline on WorkspaceBench's user_modeling family: Claude Sonnet 5.5 answers each item's chat as the assistant
(no lens, no activations), and WorkspaceBench's own judge decides whether the reply names the user attribute by inference.
This is the "what does a plain model reading the transcript get" floor the O-lens numbers should be compared against.

    source .envrc.local && uv run python -m wsb.user_modeling --n 3 --limit 10        # 5 selfdescribe + 5 synthsys, bank order
    uv run python -m wsb.user_modeling --items um-sd-country-00,um-ss-continent-02    # pick ids
    uv run python -m wsb.user_modeling --n 3 --subfamily synthsys                     # the 53 synthsys items

Run dir: data/runs/wsb_user_modeling/<date>_<slug>/ with responses.jsonl, judged.jsonl, summary.json, meta.json and an
expview run (each item = one response + its judge call); `runs.py finish` syncs it to HF at the end.
Scoring per item: pass@1 = mean over samples, pass@n = any sample; `inferred` = picked gold with basis
inferred_characterization (their headline), `correct` = picked gold, `regex` = an `intermediates` alias appears in the reply.
Responder or judge refusals count as fails in the table and are counted separately in summary.json (`refused`)."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from counterfactual_investigator_agent_with_olens.runs import finish, new_run
from wsb import judge

sys.path.insert(0, str(Path(__file__).parents[1] / "expview"))
from expview import Run, call

BANK = Path(__file__).parent / "banks" / "user_modeling.json"
RESPONDER = "claude-sonnet-5-5"
EFFORT = "medium"
METRICS = ("inferred", "correct", "regex")


def pick_items(bank: list[dict], items: str | None, limit: int, subfamily: str | None) -> list[dict]:
    if subfamily:
        bank = [it for it in bank if it["subfamily"] == subfamily]
        return bank[:limit] if limit else bank
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
    system, *turns = item["messages"]  # every item is [system, user]
    r = await judge.client.messages.create(
        model=RESPONDER,
        max_tokens=16000,
        system=system["content"],
        messages=turns,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": EFFORT},
    )
    return {
        "id": item["name"],
        "sample": sample,
        "response": "".join(b.text for b in r.content if b.type == "text"),
        "reasoning": "".join(b.thinking for b in r.content if b.type == "thinking") or None,
        "stop_reason": r.stop_reason,
        "usage": {"input": r.usage.input_tokens, "output": r.usage.output_tokens},
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
        "refused": {
            "responder": sum(r["stop_reason"] == "refusal" for r in judged),
            "judge": sum(r["pick"] == "refused" and r["stop_reason"] != "refusal" for r in judged),
        },
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
    ap.add_argument("--subfamily", choices=["selfdescribe", "synthsys"])
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--slug", default="sonnet-text-only")
    args = ap.parse_args()

    bank_file = json.loads(BANK.read_text())
    bank = bank_file["items"]
    items = pick_items(bank, args.items, args.limit, args.subfamily)
    options = judge.build_options(bank)
    run = new_run(
        "wsb_user_modeling",
        args.slug,
        description=f"WorkspaceBench user_modeling text-only baseline: {RESPONDER} answers {len(items)} items x{args.n}, WSB um-v2 judge",
        models=[
            {"name": RESPONDER, "served_by": "anthropic", "role": "responder", "effort": EFFORT, "thinking": "adaptive"},
            {"name": judge.JUDGE_MODEL, "served_by": "anthropic", "role": "judge", "effort": judge.JUDGE_EFFORT},
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
            row = await respond(item, sample)
            if row["stop_reason"] == "refusal":
                row |= {"choice": None, "basis": None, "evidence": "", "rationale": "", "pick": "refused", "regex": False}
                row |= {"judge_messages": [], "judge_output": "", "judge_usage": None}
            else:
                row |= await judge.judge(item, row["response"], opts, gold_position)
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
    (run / "responses.jsonl").write_text(
        jsonl({k: r[k] for k in ("id", "sample", "response", "reasoning", "stop_reason", "usage")} for r in judged)
    )
    (run / "judged.jsonl").write_text(jsonl({k: v for k, v in r.items() if k not in ("response", "reasoning")} for r in judged))
    summary = summarize(items, judged)
    (run / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print_table(summary, args.n)
    print("tokens:", summary["tokens"])
    finish(run)


if __name__ == "__main__":
    asyncio.run(main())
