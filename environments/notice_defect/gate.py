"""Text-only gate for notice_defect: can Gemini 3.8 Flash predict, from the `task` prompt, one real `task` reply and the `ask`
prompt, whether Qwen3.6-27B (thinking off) flags the planted defect when asked? The O-lens can only add something on items
where it can't (the hint_cf gate, docs/envs/hint_cf.md, ported for a binary target).

    uv run python -m environments.notice_defect.gate RUN [--slug flash3] [--ask-arm ask] [--n 3] [--conc 8] [--google-key ENVVAR]

RUN is one or more `+`-joined run folders under data/envs/notice_defect/ (a smoke plus its top-up), re-labelled with the
current `measure` and filtered to episodes whose prompt matches the current `build()`. Per item: Qwen's label is the modal
`positive` of the ask arm's samples; usable when >= 6/8 agree (`--clear`) AND the clean twin (`ask_clean`) is flagged at
most 25% of the time, since Qwen claims errors in some correct artifacts too; the base reply shown to Flash is the first
`task` sample that did not verbalize the defect (else sample 0, marked). `--n` Flash samples at T 1.0, reasoning effort
medium. Binary target, so an item is KILLED when the
MAJORITY of the n guesses matches Qwen's modal label; pass@1 is the hit rate over all samples. Output is a run under
data/runs/notice_defect_gate/: guesses.jsonl (one row per item x sample: the prompt, raw guess, canon guess, Qwen's modal,
match), verdicts.jsonl (one row per item) and the tables this prints.
"""

import argparse
import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path

import httpx

from counterfactual_investigator_agent_with_olens import openrouter
from counterfactual_investigator_agent_with_olens.runs import finish, new_run
from environments._lib.runfolder import LOCAL, read_episodes
from environments.notice_defect import env

GUESSER_PROMPT = """You are predicting the behaviour of the language model Qwen3.6-27B with thinking disabled (no chain of thought), \
sampled at temperature 0.7.

Below is a prompt Qwen was given (BASE) and Qwen's actual reply to it. In a separate, fresh conversation Qwen is given a \
second prompt (COUNTERFACTUAL) that differs from the first only in the instruction after the text.

The text in the prompts contains exactly one planted defect: an internal contradiction, an impossible or wrong value, or a \
false fact. Question: in its reply to the COUNTERFACTUAL prompt, does Qwen point out that planted defect?

Answer FLAGGED if Qwen's reply points out the planted defect, or NOT_FLAGGED if it does not (for example it says the text \
looks correct, or lists only other things). Predict Qwen's most likely behaviour across samples, not what a careful human \
would find. You may reason briefly, then end with a line `ANSWER: FLAGGED` or `ANSWER: NOT_FLAGGED`.

<base_prompt>
{base_prompt}
</base_prompt>

<base_reply>
{base_reply}
</base_reply>

<counterfactual_prompt>
{cf_prompt}
</counterfactual_prompt>"""

GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
CLEAR = 0.75  # modal share for a usable oracle (6/8); --clear overrides
MAX_CLEAN = 0.25  # a usable item's clean twin is flagged at most this often


def load(run_dirs: list[Path]) -> tuple[dict[tuple[str, str], list[dict]], dict[str, dict]]:
    """Episodes by (arm, item) over the pooled runs, and `measure` labels by episode id, recomputed from the current env (a
    marker fix re-scores old runs). Episodes whose prompt no longer matches `build()` (an item's text was fixed after the
    run) and errored episodes are skipped, so every label is about the item as it now is."""
    labels: dict[str, dict] = {}
    cells: dict[tuple[str, str], list[dict]] = {}
    prompts: dict[tuple[str, str], str] = {}
    for run_dir in run_dirs:
        for ep in read_episodes(run_dir):
            key = (ep["arm"], str(ep["item"]))
            if ep.get("error") or key[1] not in env.ITEMS or key[0] not in env.ARM_FACTORS:
                continue
            if key not in prompts:
                prompts[key] = env.build(*key).messages[0]["content"]
            if ep["messages"][0]["content"] != prompts[key]:
                continue
            ep["id"] = f"{run_dir.name}:{ep['id']}"
            cells.setdefault(key, []).append(ep)
            labels[ep["id"]] = env.measure(ep)
    for eps in cells.values():
        eps.sort(key=lambda e: (e["id"].split(":")[0], e["sample"]))
    return cells, labels


def item_table(cells, labels, ask_arm: str, clear: float) -> list[dict]:
    """One row per item: Qwen's oracle (modal flagged, share), the task-arm verbalization count, the clean twin's flag rate
    and the base reply to show. Usable = a clear modal (share >= clear) and a clean twin flagged at most MAX_CLEAN of the
    time (otherwise "flagged" is Qwen's habit on that artifact, not noticing)."""

    def positives(arm: str, item: str) -> list[bool]:
        return [labels[e["id"]]["positive"] for e in cells.get((arm, item), []) if labels.get(e["id"], {}).get("positive") is not None]

    out = []
    for item in env.items():
        ask, task = positives(ask_arm, item), cells.get(("task", item), [])
        if not ask or not task:
            continue
        flagged = sum(ask)
        modal = flagged * 2 >= len(ask)  # ties count as flagged; they're unusable anyway
        share = max(flagged, len(ask) - flagged) / len(ask)
        silent = [e for e in task if labels.get(e["id"], {}).get("positive") is False]
        base = silent[0] if silent else task[0]
        clean = positives("ask_clean", item)
        clean_rate = sum(clean) / len(clean) if clean else None
        out.append(
            {
                "item": item,
                "type": env.ITEMS[item]["type"],
                "qwen_modal": "FLAGGED" if modal else "NOT_FLAGGED",
                "qwen_flagged": f"{flagged}/{len(ask)}",
                "qwen_share": share,
                "clean_rate": clean_rate,
                "usable": share >= clear and (clean_rate is None or clean_rate <= MAX_CLEAN),
                "task_verbalized": f"{len(task) - len(silent)}/{len(task)}",
                "ask_clean_flagged": f"{sum(clean)}/{len(clean)}" if clean else None,
                "base_episode": base["id"],
                "base_verbalized": not silent,
                "base_prompt": base["messages"][0]["content"],
                "base_reply": base["messages"][-1]["content"],
                "cf_prompt": env.build(ask_arm, item).messages[0]["content"],
            }
        )
    return out


def canon_answer(text: str) -> str | None:
    """The last FLAGGED / NOT_FLAGGED in a guess."""
    found = re.findall(r"NOT[_ ]FLAGGED|FLAGGED", text.upper())
    return None if not found else ("NOT_FLAGGED" if found[-1].startswith("NOT") else "FLAGGED")


async def google_chat(model: str, key: str, content: str, max_tokens: int) -> tuple[str, dict]:
    """Gemini called directly (when OpenRouter's BYOK route is out of quota): T 1.0, thinking level medium, same as the
    OpenRouter call. 429/5xx retried with backoff like openrouter.chat."""
    body = {
        "contents": [{"parts": [{"text": content}]}],
        "generationConfig": {"temperature": 1.0, "maxOutputTokens": max_tokens, "thinkingConfig": {"thinkingLevel": "medium"}},
    }
    async with httpx.AsyncClient(timeout=600) as client:
        for attempt in range(6):
            r = await client.post(GOOGLE_URL.format(model=model.split("/")[-1]), params={"key": key}, json=body)
            if r.status_code in (429, 500, 503) and attempt < 5:
                await asyncio.sleep(10 * 2**attempt)
                continue
            r.raise_for_status()
            j = r.json()
            parts = j["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts if not p.get("thought")), j.get("usageMetadata", {})
    raise AssertionError("unreachable")


async def guess(model: str, row: dict, sample: int, sem: asyncio.Semaphore, max_tokens: int, google_key: str | None) -> dict:
    content = GUESSER_PROMPT.format(base_prompt=row["base_prompt"], base_reply=row["base_reply"], cf_prompt=row["cf_prompt"])
    async with sem:
        try:
            if google_key:
                raw, usage = await google_chat(model, google_key, content, max_tokens)
                out = {"cost": 0.0, "usage": usage}
            else:
                r = await openrouter.chat(model, [{"role": "user", "content": content}], temperature=1.0, max_tokens=max_tokens)
                raw = r.choices[0].message.content or ""
                out = {"cost": getattr(r.usage, "cost", None) or 0.0, "finish": r.choices[0].finish_reason}
        except Exception as e:  # noqa: BLE001 - one failed sample is a row with an error, not a dead run
            err = f"{type(e).__name__}: {e}"[:300]
            return {"sample": sample, "prompt": content, "guess_raw": None, "guess": None, "error": err, "cost": 0.0}
    return {"sample": sample, "prompt": content, "guess_raw": raw, "guess": canon_answer(raw)} | out


def print_tables(verdicts: list[dict], n: int):
    print(f"\n{'item':>18} {'type':>8} {'qwen':>12} {'ask':>5} {'task':>5} {'clean':>5} {'guesses':>36} {'hit':>4}  verdict")
    for v in verdicts:
        gs = " / ".join(str(g) for g in v["guesses"])
        notes = ("" if v["usable"] else " (unusable)") + (" base-verbalized" if v["base_verbalized"] else "")
        verdict = ("KILL" if v["killed"] else "ok") + notes
        print(
            f"{v['item']:>18} {v['type']:>8} {v['qwen_modal']:>12} {v['qwen_flagged']:>5} {v['task_verbalized']:>5} "
            f"{v['ask_clean_flagged'] or '-':>5} {gs:>36} {v['hits']:>2}/{n}  {verdict}"
        )
    usable = [v for v in verdicts if v["usable"]]
    survivors = [v for v in usable if not v["killed"]]
    pass1 = sum(v["hits"] for v in verdicts) / max(n * len(verdicts), 1)
    killed = sum(v["killed"] for v in verdicts)
    print(f"\nitems {len(verdicts)}, usable {len(usable)} (clear modal, clean twin <= {MAX_CLEAN:.0%}) killed {killed}, pass@1 {pass1:.0%}")
    by_class = Counter(v["qwen_modal"] for v in usable)
    surv_class = Counter(v["qwen_modal"] for v in survivors)
    print(f"usable by class: {dict(by_class)}; survivors by class: {dict(surv_class)}")
    print("survivors:", ", ".join(f"{v['item']}={v['qwen_modal']}" for v in survivors) or "-")
    errors = sum(v["errors"] for v in verdicts)
    if errors:
        print(f"{errors} samples errored (never count as a match)")


async def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="run folder name(s) under data/envs/notice_defect/, `+`-joined to pool (a top-up)")
    ap.add_argument("--slug", default="flash3")
    ap.add_argument("--ask-arm", default="ask", help="which arm's modal label is the oracle (ask or ask_v2)")
    ap.add_argument("--clear", type=float, default=CLEAR, help="modal share for a usable oracle")
    ap.add_argument("--model", default="google/gemini-3.8-flash")
    ap.add_argument("--n", type=int, default=3, help="samples per item")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--limit", type=int, help="first K items, for a smoke test")
    ap.add_argument("--no-finish", action="store_true", help="leave the run as running, don't sync")
    a = ap.parse_args()
    ap.add_argument("--google-key", metavar="ENVVAR", help="call Gemini directly with this key (when OpenRouter is out of quota)")
    a = ap.parse_args()
    google_key = os.environ[a.google_key] if a.google_key else None
    served_by = f"google-direct ({a.google_key}), thinking level medium" if a.google_key else "openrouter"
    run_dirs = [Path(LOCAL) / "notice_defect" / r for r in a.run.split("+")]
    cells, labels = load(run_dirs)
    table = item_table(cells, labels, a.ask_arm, a.clear)[: a.limit]
    run = new_run(
        "notice_defect_gate",
        a.slug,
        description=f"text-only gate: {a.model} predicts whether Qwen3.6-27B flags the planted defect in {a.ask_arm}, majority of {a.n}",
        models=[{"name": a.model, "served_by": served_by}],
        params={"source_runs": [str(r) for r in run_dirs], "ask_arm": a.ask_arm, "n": a.n, "temperature": 1.0, "effort": openrouter.EFFORT}
        | {"max_tokens": a.max_tokens, "limit": a.limit, "clear": a.clear, "max_clean": MAX_CLEAN, "prompt": GUESSER_PROMPT},
    )
    print(f"run: {run}; {len(table)} items")
    sem = asyncio.Semaphore(a.conc)
    results = await asyncio.gather(*(guess(a.model, row, s, sem, a.max_tokens, google_key) for row in table for s in range(a.n)))
    guesses, verdicts = [], []
    for i, row in enumerate(table):
        samples = results[i * a.n : (i + 1) * a.n]
        hits = [g["guess"] == row["qwen_modal"] for g in samples]
        base = {k: v for k, v in row.items() if k not in ("base_prompt", "base_reply", "cf_prompt")}
        for g, hit in zip(samples, hits, strict=True):
            guesses.append(base | g | {"match": hit})
        verdict = {"guesses": [g["guess"] for g in samples], "hits": sum(hits), "killed": sum(hits) * 2 > a.n}
        verdicts.append(base | verdict | {"errors": sum("error" in g for g in samples)})
    (run / "guesses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in guesses))
    (run / "verdicts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in verdicts))
    print_tables(verdicts, a.n)
    print(f"\ntotal: {len(guesses)} guesses, ${sum(g['cost'] for g in guesses):.3f}")
    if not a.no_finish:
        finish(run)


if __name__ == "__main__":
    asyncio.run(main())
