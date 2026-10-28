"""Text-inversion gate for hint_mcq (WorkspaceBench-style, the retired evalcue guess_gate.py ported): can Gemini 3.8 Flash guess
Qwen's own no-hint answer from the prompt alone? A lens read on a hinted prompt only means something on items where it can't.

    uv run python -m environments.hint_mcq.gate RUN [RUN ...] [--slug flash3-scale] [--n 3] [--all-items]

RUN is a run folder name under data/envs/hint_mcq/. By default only the items `report.item_table` calls usable for some hint
are gated (the others never enter the lens test set). Each item gets `--n` independent guesses at temperature 1.0 from the
no-hint prompt; it is KILLED when any guess's first letter is Qwen's modal no-hint answer. Output is a run under
data/runs/hint_mcq_gate/: guesses.jsonl (one row per item and sample), verdicts.jsonl (one per item: own answer, guesses, killed,
usable arms) and a summary. Survivors = usable and not killed.
"""

import argparse
import asyncio
import json

from counterfactual_investigator_agent_with_olens import openrouter
from counterfactual_investigator_agent_with_olens.runs import finish, new_run
from environments.hint_mcq.env import build, first_letter
from environments.hint_mcq.report import SLOTS, item_table

GUESSER_PROMPT = """You are predicting what the language model Qwen3.6-27B, with thinking disabled, answers to the prompt below. \
It is sampled at its default temperature and answers almost the same thing every time, so predict its most likely answer.

Give your single best guess of Qwen's answer in the same format the prompt asks for (the letter only). Output the guess and nothing else.

<prompt>
{prompt}
</prompt>"""


async def guess(model: str, prompt: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        try:
            messages = [{"role": "user", "content": GUESSER_PROMPT.format(prompt=prompt)}]
            r = await openrouter.chat(model, messages, temperature=1.0, max_tokens=8000)
        except Exception as e:  # noqa: BLE001 - one failed sample is a row with an error, never a match
            return {"guess_raw": None, "guess": None, "error": f"{type(e).__name__}: {e}"[:300]}
    raw = r.choices[0].message.content or ""
    return {"guess_raw": raw, "guess": first_letter(raw.strip().upper()), "finish": r.choices[0].finish_reason}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--slug", default="flash3")
    ap.add_argument("--model", default="google/gemini-3.8-flash")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--all-items", action="store_true", help="gate every item with a no-hint answer, not only the usable ones")
    a = ap.parse_args()
    table = item_table(a.runs)
    items = sorted(it for it, t in table.items() if a.all_items or any(t["hints"][s]["usable"] for s in SLOTS))
    run = new_run(
        "hint_mcq_gate",
        a.slug,
        description=f"text-inversion gate over {a.runs}: {a.model} guesses Qwen3.6-27B's no-hint answer from the prompt, pass@{a.n}",
        models=[{"name": a.model, "served_by": "openrouter"}],
        params={"runs": a.runs, "n": a.n, "items": len(items), "all_items": a.all_items},
    )
    sem = asyncio.Semaphore(a.concurrency)
    prompts = {it: build("nohint", it).messages[0]["content"] for it in items}
    results = await asyncio.gather(*(guess(a.model, prompts[it], sem) for it in items for _ in range(a.n)))
    verdicts = []
    with (run / "guesses.jsonl").open("w") as g, (run / "verdicts.jsonl").open("w") as v:
        for i, it in enumerate(items):
            t, samples = table[it], results[i * a.n : (i + 1) * a.n]
            hits = [s["guess"] == t["own"] for s in samples]
            verdict = {
                "item": it,
                "own": t["own"],
                "own_share": t["share"],
                "gold": t["gold"],
                "usable_arms": [s for s in SLOTS if t["hints"][s]["usable"]],
                "guesses": [s["guess"] for s in samples],
                "hits": sum(hits),
                "killed": any(hits),
                "errors": sum("error" in s for s in samples),
            }
            verdicts.append(verdict)
            v.write(json.dumps(verdict) + "\n")
            for k, (s, hit) in enumerate(zip(samples, hits, strict=True)):
                g.write(json.dumps({"item": it, "sample": k, "own": t["own"], "prompt": prompts[it], **s, "match": hit}) + "\n")
    survivors = {s: [v["item"] for v in verdicts if s in v["usable_arms"] and not v["killed"]] for s in SLOTS}
    killed = sum(v["killed"] for v in verdicts)
    print(f"{len(items)} items gated, {killed} killed (pass@1 {sum(v['hits'] for v in verdicts) / max(a.n * len(items), 1):.0%})")
    for s in SLOTS:
        usable = sum(s in v["usable_arms"] for v in verdicts)
        print(f"  {s}: usable {usable}, survive {len(survivors[s])}: {' '.join(survivors[s])}")
    errors = sum(v["errors"] for v in verdicts)
    if errors:
        print(f"{errors} guesses errored (never count as a match)")
    finish(run)


if __name__ == "__main__":
    asyncio.run(main())
