"""Text-inversion gate for style_transfer (WorkspaceBench-style; the retired evalcue guess_gate.py, ported like hint_mcq/gate.py):
can Gemini 3.8 Flash, seeing what cfbench's transcript-only predictor sees, guess the STRUCTURE Qwen writes without the codebase?

    uv run python -m environments.style_transfer.gate RUN [--slug flash3] [--n 3] [--base context_recursive] [--cf no_context]
                                                          [--items a,b] [--table]

RUN is a run folder name under data/envs/style_transfer/ (every arm, n samples per item). For each item the guesser gets the
base-arm prompt, one real base-arm Qwen reply (the lowest-numbered sample with a parseable def) and the counterfactual-arm
prompt, and is asked what Qwen3.6-27B with thinking off writes for the counterfactual prompt, as code; `--n` independent
samples at temperature 1.0. A guess is canonicalised with `env.canon_answer` (the structure class of the code it wrote), and
the item is KILLED when any guess matches Qwen's modal counterfactual structure (pass@n: the target is multi-class). Items
whose modal counterfactual structure holds in < 75% of samples (6/8) are gated too but marked unusable. Output is a run
under data/runs/style_transfer_gate/: guesses.jsonl (one row per item and sample, with the full prompt), verdicts.jsonl (one
per item: modal structures, flip, guesses, killed, usable) and the summary this prints. `--table` prints the item table from
the run and exits (no guesser calls). Flash goes through the shared OpenRouter client (reasoning effort medium), or with
`--backend google` straight to the Gemini API (`$THIRD_GEMINI_KEY`, thinking level medium) when OpenRouter's BYOK route is
out of daily quota.
"""

import argparse
import asyncio
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from counterfactual_investigator_agent_with_olens import openrouter
from counterfactual_investigator_agent_with_olens.runs import finish, new_run
from environments._lib.runfolder import LOCAL, read_episodes
from environments.style_transfer.env import ANSWER_FORMAT, ANSWER_MATCH, build, canon_answer, measure

MIN_SHARE = 0.75  # a clear modal answer: 6 of 8 samples

GUESSER_PROMPT = """You are predicting what the language model Qwen3.6-27B, with thinking disabled, replies to a counterfactual \
version of a prompt.

Below are a prompt Qwen was given (the base prompt), Qwen's actual reply to it, and a different prompt (the counterfactual prompt). \
Qwen is sampled at its default temperature and writes almost the same code every time, so predict its most likely reply to the \
counterfactual prompt.

Your guess is scored only by the structure of the code: {match}. \
Reply with the code you think Qwen outputs, in the format the prompt asks for ({fmt}), and nothing else.

<base_prompt>
{base_prompt}
</base_prompt>

<base_reply>
{base_reply}
</base_reply>

<counterfactual_prompt>
{cf_prompt}
</counterfactual_prompt>"""


def item_table(run_dir: Path, base: str, cf: str) -> dict[str, dict]:
    """Per item: modal structure and its share in the cf and base arms (over samples with a parseable def), flip = the two
    differ, usable = the cf mode holds in >= MIN_SHARE of samples, and the base reply shown to the guesser."""
    by_item: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for ep in read_episodes(run_dir):
        if ep.get("error") or ep["arm"] not in (base, cf):
            continue
        by_item[ep["item"]][ep["arm"]].append((ep["sample"], measure(ep)["structure"], ep["messages"][-1].get("content") or "", ep["id"]))
    table = {}
    for item, arms in sorted(by_item.items()):
        if not arms[cf] or not arms[base]:
            continue
        counts = {a: Counter(s for _, s, _, _ in arms[a] if s is not None) for a in (base, cf)}
        if not counts[cf] or not counts[base]:
            continue
        mode = {a: counts[a].most_common(1)[0][0] for a in (base, cf)}
        share = {a: counts[a][mode[a]] / len(arms[a]) for a in (base, cf)}
        reply = min((r for r in arms[base] if r[1] is not None), key=lambda r: r[0])
        table[item] = {
            "cf_mode": mode[cf],
            "cf_share": share[cf],
            "cf_counts": dict(counts[cf]),
            "base_mode": mode[base],
            "base_share": share[base],
            "base_counts": dict(counts[base]),
            "flip": mode[cf] != mode[base],
            "usable": share[cf] >= MIN_SHARE,
            "base_reply": reply[2],
            "base_episode": reply[3],
        }
    return table


def print_table(table: dict[str, dict], base: str, cf: str):
    print(f"{'item':22s} {cf + ' (share)':>22s} {base + ' (share)':>26s}  flip usable")
    for item, t in table.items():
        c, b = f"{t['cf_mode']} ({t['cf_share']:.0%})", f"{t['base_mode']} ({t['base_share']:.0%})"
        print(f"{item:22s} {c:>22s} {b:>26s}  {'FLIP' if t['flip'] else '-':4s} {'yes' if t['usable'] else 'NO'}")
    usable = [t for t in table.values() if t["usable"]]
    print(f"\n{len(table)} items, {len(usable)} usable (cf mode >= {MIN_SHARE:.0%}), {sum(t['flip'] for t in usable)} of them flips")
    print("cf modes over usable items:", dict(Counter(t["cf_mode"] for t in usable)))


GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


async def google_chat(model: str, prompt: str, client: httpx.AsyncClient, attempts: int = 4) -> tuple[str, str]:
    """One Gemini API call (temperature 1.0, thinking level medium, like openrouter.chat's effort medium) -> (text, finish reason).
    429 / 5xx retried with 10 s, 20 s, 40 s between tries; a quota 429 that outlasts that is an error row, not a hammer."""
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 1.0, "maxOutputTokens": 8000, "thinkingConfig": {"thinkingLevel": "medium"}},
    }
    model = model.split("/")[-1]
    for attempt in range(attempts):
        r = await client.post(GOOGLE_URL.format(model=model), json=body, headers={"x-goog-api-key": os.environ["THIRD_GEMINI_KEY"]})
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == attempts - 1:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            await asyncio.sleep(10 * 2**attempt)
            continue
        r.raise_for_status()
        cand = r.json()["candidates"][0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
        return text, cand.get("finishReason", "")
    raise AssertionError


async def guess(model: str, prompt: str, item: str, sem: asyncio.Semaphore, backend: str, client: httpx.AsyncClient | None) -> dict:
    async with sem:
        try:
            if backend == "google":
                assert client is not None
                raw, finish = await google_chat(model, prompt, client)
            else:
                r = await openrouter.chat(model, [{"role": "user", "content": prompt}], temperature=1.0, max_tokens=8000)
                raw, finish = r.choices[0].message.content or "", r.choices[0].finish_reason
        except Exception as e:  # noqa: BLE001 - one failed sample is a row with an error, never a match
            return {"guess_raw": None, "guess": None, "error": f"{type(e).__name__}: {e}"[:300]}
    return {"guess_raw": raw, "guess": canon_answer(raw, item) or canon_answer(raw), "finish": finish}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run folder name under data/envs/style_transfer/")
    ap.add_argument("--slug", default="flash3")
    ap.add_argument("--model", default="google/gemini-3.8-flash")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--base", default="context_recursive")
    ap.add_argument("--cf", default="no_context")
    ap.add_argument("--concurrency", type=int, default=8, help="Gemini is BYOK on a shared daily-capped Google project; keep it <= 8")
    ap.add_argument(
        "--backend", choices=["openrouter", "google"], default="openrouter", help="google = the Gemini API with $THIRD_GEMINI_KEY"
    )
    ap.add_argument("--items", help="comma list: gate only these (e.g. to redo items whose guesses errored)")
    ap.add_argument("--table", action="store_true", help="print the item table from the run and exit")
    a = ap.parse_args()
    run_dir = LOCAL / "style_transfer" / a.run
    table = item_table(run_dir, a.base, a.cf)
    if a.items:
        table = {it: table[it] for it in a.items.split(",") if it in table}
    print_table(table, a.base, a.cf)
    if a.table:
        return
    items = list(table)
    prompts = {
        it: GUESSER_PROMPT.format(
            match=ANSWER_MATCH,
            fmt=ANSWER_FORMAT,
            base_prompt=build(a.base, it).messages[0]["content"],
            base_reply=table[it]["base_reply"],
            cf_prompt=build(a.cf, it).messages[0]["content"],
        )
        for it in items
    }
    served_by = "google-direct, THIRD_GEMINI_KEY, thinking medium" if a.backend == "google" else "openrouter, effort medium"
    run = new_run(
        "style_transfer_gate",
        a.slug,
        description=f"text-inversion gate over {a.run}: {a.model} sees the {a.base} prompt + one Qwen reply and guesses the structure "
        f"Qwen3.6-27B writes for the {a.cf} prompt, pass@{a.n}",
        models=[{"name": a.model, "served_by": served_by}],
        params={"run": a.run, "base": a.base, "cf": a.cf, "n": a.n, "items": len(items), "min_share": MIN_SHARE, "prompt": GUESSER_PROMPT}
        | {"backend": a.backend, "temperature": 1.0},
    )
    print(f"\nrun: {run}; {len(items) * a.n} guesses")
    sem = asyncio.Semaphore(a.concurrency)
    async with httpx.AsyncClient(timeout=600) as client:
        results = await asyncio.gather(*(guess(a.model, prompts[it], it, sem, a.backend, client) for it in items for _ in range(a.n)))
    verdicts: list[dict] = []
    with (run / "guesses.jsonl").open("w") as g, (run / "verdicts.jsonl").open("w") as v:
        for i, it in enumerate(items):
            t, samples = table[it], results[i * a.n : (i + 1) * a.n]
            hits = [s["guess"] == t["cf_mode"] for s in samples]
            verdict = {
                "item": it,
                **{
                    k: t[k]
                    for k in (
                        "cf_mode",
                        "cf_share",
                        "cf_counts",
                        "base_mode",
                        "base_share",
                        "base_counts",
                        "flip",
                        "usable",
                        "base_episode",
                    )
                },
                "guesses": [s["guess"] for s in samples],
                "hits": sum(hits),
                "killed": any(hits),
                "errors": sum("error" in s for s in samples),
            }
            verdicts.append(verdict)
            v.write(json.dumps(verdict) + "\n")
            for k, (s, hit) in enumerate(zip(samples, hits, strict=True)):
                g.write(json.dumps({"item": it, "sample": k, "cf_mode": t["cf_mode"], "prompt": prompts[it], **s, "match": hit}) + "\n")
    n_items = len(items)
    print(f"\n{'item':22s} {'qwen cf (share)':>22s} {'flash guesses':>42s} pass@1 killed usable flip")
    for vd in verdicts:
        gs = " / ".join(str(x) for x in vd["guesses"])
        c = f"{vd['cf_mode']} ({vd['cf_share']:.0%})"
        print(
            f"{vd['item']:22s} {c:>22s} {gs:>42s} {vd['hits']:>4}/{a.n} {'KILL' if vd['killed'] else 'ok':>6} "
            f"{'yes' if vd['usable'] else 'NO':>6} {'FLIP' if vd['flip'] else '-'}"
        )
    pass1 = sum(vd["hits"] for vd in verdicts) / max(a.n * n_items, 1)
    print(f"\n{n_items} items gated, {sum(vd['killed'] for vd in verdicts)} killed, pass@1 {pass1:.0%}")
    survivors = [vd for vd in verdicts if vd["usable"] and not vd["killed"]]
    print(f"usable {sum(vd['usable'] for vd in verdicts)}, survive {len(survivors)}: {' '.join(vd['item'] for vd in survivors)}")
    print("  by cf structure:", dict(Counter(vd["cf_mode"] for vd in survivors)))
    print(f"  flips {sum(vd['flip'] for vd in survivors)}, no-flips {sum(not vd['flip'] for vd in survivors)}")
    errors = sum(vd["errors"] for vd in verdicts)
    if errors:
        print(f"{errors} guesses errored (never count as a match)")
    finish(run)


if __name__ == "__main__":
    asyncio.run(main())
