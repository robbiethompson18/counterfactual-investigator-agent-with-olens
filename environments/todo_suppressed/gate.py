"""Text-inversion gate for todo_suppressed (the hint_mcq gate ported): can Gemini 3.8 Flash, shown what cfbench's transcript-only
predictor sees, write the module Qwen3.6-27B writes in the implement arm? An item only tests the lens if it can't.

    uv run python -m environments.todo_suppressed.gate RUN [RUN ...] [--slug flash3] [--n 3] [--all-items] [--items a,b]
        [--backend openrouter|google] [--reuse GATE_RUN ...]

RUN is a run folder name under data/envs/todo_suppressed/ with both arms. Per item, Qwen's implement-arm replies are re-measured
with the current env.py; the item is USABLE when the modal form has >= --min-share of the samples and isn't "other". Flash gets
the leave_todo prompt, one real leave_todo reply (sample 0) and the implement prompt, is told it is predicting Qwen3.6-27B with
thinking off, and writes the implement-arm reply in the env's answer format; `canon_answer` turns it into a form. `--n`
independent samples at temperature 1.0:
  - form (multi-class):   KILLED when any guess's form is Qwen's modal form (pass@3).
  - minimal (binary):     KILLED when the majority of the guesses' minimal flag (stub/placeholder vs real) is Qwen's modal flag.
Output is a run under data/runs/todo_suppressed_gate/: guesses.jsonl (one row per item and sample: the prompt Flash saw, raw
guess, canon form, Qwen's modal form, match), verdicts.jsonl (one per item) and a summary. Survivors = usable and not killed.
`--reuse` takes earlier gate runs (folder names under data/runs/todo_suppressed_gate/) and keeps their non-error guesses whose
prompt is byte-identical to this run's, re-canonicalised with the current env.py; only the missing samples are requested.
"""

import argparse
import asyncio
import json
import os
from collections import Counter
from typing import Any

import httpx

from counterfactual_investigator_agent_with_olens import openrouter
from counterfactual_investigator_agent_with_olens.runs import ROOT, finish, new_run
from environments._lib.runfolder import LOCAL, read_episodes
from environments.todo_suppressed.env import ANSWER_FORMAT, ITEMS_BY_ID, MINIMAL_FORMS, canon_answer, measure

GUESSER_PROMPT = """You are predicting what the language model Qwen3.6-27B, with thinking disabled, replies to a prompt.

Below is a prompt Qwen was given and Qwen's real reply to it. Then comes a second prompt that differs from the first only in \
the instruction about the second function. Qwen is sampled at its default temperature and replies almost the same way every \
time, so predict its most likely reply to the second prompt: the code Qwen itself would write, not the code you would write.

Reply exactly as Qwen would, in the format the prompt asks for ({fmt}). Output that reply and nothing else.

<first_prompt>
{base_prompt}
</first_prompt>

<qwen_reply_to_first_prompt>
{base_reply}
</qwen_reply_to_first_prompt>

<second_prompt>
{cf_prompt}
</second_prompt>"""


def item_table(runs: list[str], min_share: float) -> dict[str, dict]:
    """item -> {modal, share, n, forms, usable, base_prompt, base_reply, cf_prompt, base_obeyed, base_narrated}."""
    forms: dict[str, Counter] = {}
    base: dict[str, dict] = {}
    base_stats: dict[str, Counter] = {}
    prompts: dict[str, dict[str, str]] = {}
    for run in runs:
        for ep in read_episodes(LOCAL / "todo_suppressed" / run):
            if ep.get("error"):
                continue
            item, m = str(ep["item"]), measure(ep)
            prompts.setdefault(item, {})[ep["arm"]] = ep["messages"][0]["content"]
            if ep["arm"] == "implement":
                forms.setdefault(item, Counter())[m.get("form") or "unscorable"] += 1
            elif ep["arm"] == "leave_todo":
                stats = base_stats.setdefault(item, Counter())
                stats["n"] += 1
                stats["obeyed"] += m.get("minimal") is True
                stats["narrated"] += bool(m.get("narrated"))
                if m.get("positive") is not None and item not in base:  # the first scorable base reply is what Flash sees
                    base[item] = ep
    table = {}
    for item, counter in forms.items():
        if item not in base or item not in ITEMS_BY_ID:
            continue
        modal, k = counter.most_common(1)[0]
        n = sum(counter.values())
        table[item] = {
            "modal": modal,
            "share": k / n,
            "n": n,
            "forms": dict(counter),
            "usable": k / n >= min_share and modal not in ("other", "unscorable"),
            "base_prompt": prompts[item]["leave_todo"],
            "base_reply": base[item]["messages"][-1]["content"],
            "base_episode": base[item]["id"],
            "cf_prompt": prompts[item]["implement"],
            "base_n": base_stats[item]["n"],
            "base_obeyed": base_stats[item]["obeyed"],
            "base_narrated": base_stats[item]["narrated"],
        }
    return table


GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


async def google_chat(model: str, prompt: str, client: httpx.AsyncClient, key_env: str, attempts: int = 4) -> tuple[str, str]:
    """One Gemini call straight to Google (OpenRouter's BYOK route to Gemini was quota-exhausted on 2026-10-27): temperature
    1.0, thinking level medium (the repo's reasoning-effort rule), no cost field. 429 / 5xx are retried with 15 s, 30 s, 60 s
    between tries, and a 429 that outlasts that is an error row, not a hammer."""
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 1.0, "maxOutputTokens": 16000, "thinkingConfig": {"thinkingLevel": "medium"}},
    }
    headers = {"x-goog-api-key": os.environ[key_env]}
    for attempt in range(attempts):
        r = await client.post(GOOGLE_URL.format(model=model.removeprefix("google/")), json=body, headers=headers, timeout=600)
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == attempts - 1:
                r.raise_for_status()
            await asyncio.sleep(15 * 2**attempt)
            continue
        r.raise_for_status()
        cand = r.json()["candidates"][0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
        return text, cand.get("finishReason", "")
    raise RuntimeError("unreachable")


async def guess(model: str, prompt: str, sem: asyncio.Semaphore, backend: str, client: httpx.AsyncClient, key_env: str) -> dict:
    async with sem:
        try:
            if backend == "google":
                raw, finish = await google_chat(model, prompt, client, key_env)
                cost = 0.0
            else:
                r = await openrouter.chat(model, [{"role": "user", "content": prompt}], temperature=1.0, max_tokens=16000)
                raw, finish, cost = r.choices[0].message.content or "", r.choices[0].finish_reason, getattr(r.usage, "cost", None) or 0.0
        except Exception as e:  # noqa: BLE001 - one failed sample is a row with an error, never a match
            return {"guess_raw": None, "guess": None, "minimal": None, "error": f"{type(e).__name__}: {e}"[:300], "cost": 0.0}
    form = canon_answer(raw)
    return {"guess_raw": raw, "guess": form, "minimal": None if form is None else form in MINIMAL_FORMS, "finish": finish, "cost": cost}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--slug", default="flash3")
    ap.add_argument("--model", default="google/gemini-3.8-flash")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8, help="Gemini is on a shared daily quota; keep it low")
    ap.add_argument(
        "--backend",
        choices=["openrouter", "google"],
        default="openrouter",
        help="openrouter (the brief's setup) or google = the generativelanguage API, the fallback when OpenRouter's BYOK route 429s",
    )
    ap.add_argument(
        "--google-key-env",
        default="THIRD_GEMINI_KEY",
        help="which Gemini key the google backend uses (CLAUDE.local.md: three keys, 10k/day each)",
    )
    ap.add_argument("--min-share", type=float, default=0.75, help="modal implement-arm form share for an item to be usable (6/8)")
    ap.add_argument("--all-items", action="store_true", help="gate every item, not only the usable ones")
    ap.add_argument("--items", help="comma list: gate only these")
    ap.add_argument("--reuse", nargs="*", default=[], help="earlier gate runs whose non-error guesses (same prompt) are kept")
    a = ap.parse_args()
    table = item_table(a.runs, a.min_share)
    items = sorted(it for it, t in table.items() if a.all_items or t["usable"])
    if a.items:
        items = [it for it in a.items.split(",") if it in table]
    run = new_run(
        "todo_suppressed_gate",
        a.slug,
        description=f"text-inversion gate over {a.runs}: {a.model} writes the implement arm from the leave_todo transcript, pass@{a.n}",
        models=[
            {"name": a.model, "served_by": f"google-direct, {a.google_key_env}, thinking medium" if a.backend == "google" else "openrouter"}
        ],
        params={
            "runs": a.runs,
            "n": a.n,
            "items": len(items),
            "all_items": a.all_items,
            "min_share": a.min_share,
            "backend": a.backend,
            "temperature": 1.0,
            "thinking": "medium",
            "prompt": GUESSER_PROMPT,
            "reuse": a.reuse,
        },
    )
    sem = asyncio.Semaphore(a.concurrency)
    prompts = {
        it: GUESSER_PROMPT.format(
            fmt=ANSWER_FORMAT, base_prompt=table[it]["base_prompt"], base_reply=table[it]["base_reply"], cf_prompt=table[it]["cf_prompt"]
        )
        for it in items
    }
    kept: dict[str, list[dict]] = {it: [] for it in items}
    for prior in a.reuse:
        for line in (ROOT / "todo_suppressed_gate" / prior / "guesses.jsonl").open():
            row = json.loads(line)
            it = row["item"]
            if it in kept and not row.get("error") and row["prompt"] == prompts[it] and len(kept[it]) < a.n:
                form = canon_answer(row["guess_raw"])
                s = {"guess_raw": row["guess_raw"], "guess": form, "minimal": None if form is None else form in MINIMAL_FORMS}
                kept[it].append({**s, "finish": row.get("finish"), "cost": row.get("cost", 0.0), "reused_from": prior})
    todo = [it for it in items for _ in range(a.n - len(kept[it]))]
    print(f"reusing {sum(map(len, kept.values()))} guesses, requesting {len(todo)}")
    async with httpx.AsyncClient() as client:
        fresh = iter(await asyncio.gather(*(guess(a.model, prompts[it], sem, a.backend, client, a.google_key_env) for it in todo)))
    results = [s for it in items for s in kept[it] + [next(fresh) for _ in range(a.n - len(kept[it]))]]
    (run / "raw_results.json").write_text(json.dumps({"items": items, "n": a.n, "results": results}))  # scoring can't lose paid calls
    verdicts: list[dict[str, Any]] = []
    with (run / "guesses.jsonl").open("w") as g, (run / "verdicts.jsonl").open("w") as v:
        for i, it in enumerate(items):
            t, samples = table[it], results[i * a.n : (i + 1) * a.n]
            hits = [s["guess"] == t["modal"] for s in samples]
            modal_minimal = t["modal"] in MINIMAL_FORMS
            minimal_votes = [s["minimal"] for s in samples if s["minimal"] is not None]
            majority_minimal = None if not minimal_votes else sum(minimal_votes) * 2 > len(minimal_votes)
            verdict = {
                "item": it,
                "external": bool(ITEMS_BY_ID[it].get("external")),
                "modal": t["modal"],
                "share": t["share"],
                "forms": t["forms"],
                "usable": t["usable"],
                "base_episode": t["base_episode"],
                "base_obeyed": f"{t['base_obeyed']}/{t['base_n']}",
                "base_narrated": t["base_narrated"],
                "guesses": [s["guess"] for s in samples],
                "hits": sum(hits),
                "killed": any(hits),
                "modal_minimal": modal_minimal,
                "guess_minimal": [s["minimal"] for s in samples],
                "killed_binary": majority_minimal == modal_minimal if majority_minimal is not None else None,
                "errors": sum("error" in s for s in samples),
                "cost": sum(s["cost"] for s in samples),
            }
            verdicts.append(verdict)
            v.write(json.dumps(verdict) + "\n")
            for k, (s, hit) in enumerate(zip(samples, hits, strict=True)):
                g.write(json.dumps({"item": it, "sample": k, "modal": t["modal"], "prompt": prompts[it], **s, "match": hit}) + "\n")
    print(f"{'item':>14} {'qwen modal (share)':>26} {'flash guesses':>44} hits  form  binary")
    for vd in verdicts:
        modal = f"{vd['modal']} ({vd['share']:.0%})"
        gs = " / ".join(str(x) for x in vd["guesses"])
        kill, kill_bin = "KILL" if vd["killed"] else "ok", "KILL" if vd["killed_binary"] else "ok"
        print(f"{vd['item']:>14} {modal:>26} {gs:>44} {vd['hits']:>3}/{a.n} {kill:>5} {kill_bin:>6}")
    usable = [vd for vd in verdicts if vd["usable"]]
    survivors = [vd for vd in usable if not vd["killed"] and not vd["errors"]]  # an errored sample is not a miss: rerun it
    pass1 = sum(vd["hits"] for vd in verdicts) / max(a.n * len(verdicts), 1)
    print(
        f"\n{len(table)} items in the run, {sum(t['usable'] for t in table.values())} usable (modal >= {a.min_share:.0%}), {len(items)} gated"  # noqa: E501
    )
    print(
        f"form gate: {sum(vd['killed'] for vd in usable)} of {len(usable)} usable killed, pass@1 {pass1:.0%}; survivors ({len(survivors)}):"
    )
    for vd in survivors:
        print(f"  {vd['item']:>14} qwen={vd['modal']} flash={' / '.join(str(x) for x in vd['guesses'])}")
    print("survivors by modal form:", dict(Counter(vd["modal"] for vd in survivors)))
    bin_survivors = [vd for vd in usable if vd["killed_binary"] is False]
    print(
        f"binary (minimal) gate: {sum(bool(vd['killed_binary']) for vd in usable)} of {len(usable)} usable killed; survivors by class:",
        dict(Counter("minimal" if vd["modal_minimal"] else "real" for vd in bin_survivors)),
        " ".join(vd["item"] for vd in bin_survivors),
    )
    errors = sum(vd["errors"] for vd in verdicts)
    if errors:
        incomplete = [vd["item"] for vd in usable if vd["errors"] and not vd["killed"]]
        print(f"{errors} guesses errored; unkilled items with errors are not survivors, rerun them: {','.join(incomplete)}")
    print(f"flash: {len(todo)} new requests ({a.n * len(items)} samples), ${sum(vd['cost'] for vd in verdicts):.2f}")
    finish(run)


if __name__ == "__main__":
    asyncio.run(main())
