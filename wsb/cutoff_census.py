"""Which investigator shares Qwen's staleness? Ask a candidate model every knowledge_cutoff question directly (no system prompt,
one sample, reasoning effort medium) and count how often its own answer is Qwen's stale answer, the truth, or neither
(Robbie, 2026-10-29: an investigator that holds the same stale fact as the subject predicts it for free, so pick the
investigator for this family by this census, not by its model card).

    source .envrc.local && uv run python -m wsb.cutoff_census --models anthropic/claude-haiku-5.5,openai/gpt-6-luna

One run per model, data/runs/wsb_cutoff_investigator/<date>_census-<model slug>/: responses.jsonl, summary.json (rates overall
and by label_type), expview/ (one item per question); `runs.py finish` syncs it to HF and Docent.
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from counterfactual_investigator_agent_with_olens import openrouter
from counterfactual_investigator_agent_with_olens.runs import finish, new_run
from wsb import banks
from wsb.knowledge_cutoff import OptionForms

sys.path.insert(0, str(Path(__file__).parents[1] / "expview"))
from expview import Run, call

DEFAULT_MODELS = "google/gemini-3.8-flash,anthropic/claude-haiku-5.5,anthropic/claude-sonnet-5.5,openai/gpt-6.1-terra,openai/gpt-6-luna"


def verdict(item: dict, text: str) -> dict:
    named = OptionForms.build(item["option_units"]).named(text)
    truth = (item.get("truth") or {}).get("canonical")
    return {"named": sorted(named), "qwen": item["gold"] in named, "truth": truth in named if truth else None}


async def census(model: str, items: list[dict], effort: str, concurrency: int):
    slug = "census-" + model.split("/")[-1].replace(".", "")
    run = new_run(
        "wsb_cutoff_investigator",
        slug,
        description=f"knowledge_cutoff census: {model} asked each of the {len(items)} questions directly (effort {effort}, 1 sample): "
        "does its own answer equal Qwen's stale answer, the truth, or neither?",
        models=[{"name": model, "served_by": "openrouter", "effort": effort}],
        params={"effort": effort, "items": [it["id"] for it in items]},
    )
    slots = asyncio.Semaphore(concurrency)

    async def one(item: dict) -> dict:
        messages = [{"role": "user", "content": item["user"]}]
        async with slots:
            r = await openrouter.chat(model, messages, effort=effort, max_tokens=4000)
        text = (r.choices[0].message.content or "").strip()
        return {"id": item["id"], "response": text, "sent": messages, "cost": getattr(r.usage, "cost", 0) or 0, **verdict(item, text)}

    rows = await asyncio.gather(*(one(it) for it in items))
    view = Run(run / "expview", title=f"knowledge_cutoff census: {model}", description=__doc__, label_order=["qwen", "truth", "label_type"])
    for it, r in zip(items, rows, strict=True):
        view.add(
            id=it["id"],
            labels={"qwen": r["qwen"], "truth": r["truth"], "label_type": it["label_type"]},
            calls=[call("answer", model, r["sent"], output={"role": "assistant", "content": r["response"]}, parsed={"named": r["named"]})],
            extra={"gold": it["gold"], "truth": (it.get("truth") or {}).get("canonical"), "question": it["question"]},
        )
    keep = ("id", "response", "named", "qwen", "truth", "cost")
    (run / "responses.jsonl").write_text("".join(json.dumps({k: r[k] for k in keep}, ensure_ascii=False) + "\n" for r in rows))

    def rates(sub: list[dict]) -> dict:
        n = len(sub)
        return {"n": n, "qwen": sum(r["qwen"] for r in sub) / n, "truth": sum(bool(r["truth"]) for r in sub) / n,
                "neither": sum(not r["qwen"] and not r["truth"] for r in sub) / n}  # fmt: skip

    types = sorted(Counter(it["label_type"] for it in items))
    by_type = {t: rates([r for r, it in zip(rows, items, strict=True) if it["label_type"] == t]) for t in types}
    summary = {"model": model, "overall": rates(rows), "by_label_type": by_type, "cost": sum(r["cost"] for r in rows)}
    (run / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    o = summary["overall"]
    print(f"{model:32} qwen {o['qwen']:.2f}  truth {o['truth']:.2f}  neither {o['neither']:.2f}  ${summary['cost']:.2f}")
    finish(run)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8, help="per model")
    args = ap.parse_args()
    items = [it for it in banks.load("knowledge_cutoff")["items"] if not it.get("g3_leak")][: args.limit or None]
    await asyncio.gather(*(census(m, items, args.effort, args.concurrency) for m in args.models.split(",")))


if __name__ == "__main__":
    asyncio.run(main())
