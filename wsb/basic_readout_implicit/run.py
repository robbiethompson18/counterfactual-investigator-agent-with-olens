"""Text-only baseline for WorkspaceBench's basic_readout implicit items: can Gemini 3.8 Flash, told to answer as Qwen3.6-27B,
guess Qwen's own one-word favourites? Graded with WorkspaceBench's own bank judge (judge.py) and compared with the oracle
lens's per-item verdicts from agu18dec/olens_ablations (an RL ablation arm judged by Opus, not the published checkpoint).

    source .envrc.local && uv run python -m wsb.basic_readout_implicit.run --draws 5            # 160 reader + 160 judge calls
    uv run python -m wsb.basic_readout_implicit.run --limit 3 --draws 2 --no-sync               # smoke

Google rejects a request ending in an assistant turn ("Requests ending with a model turn are not supported", both via
OpenRouter and direct, checked 2026-10-28), so the stem "My one-word answer is:" goes at the end of the user turn as
`Complete this: "..."` instead of an assistant prefill. Every call goes to OpenRouter first and falls back to Google's API
with $GEMINI_API_KEY when OpenRouter errors (402/403/429/timeout); each call records its route and key name.
Writes data/runs/wsb_basic_readout/<date>_<slug>/: responses.jsonl (one line per draw with reader + judge I/O), lens.json,
summary.json (the per-item table and bootstrap CIs), expview/ (every call), meta.json; then syncs to HF (runs.py)."""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import openai
from huggingface_hub import hf_hub_download

from counterfactual_investigator_agent_with_olens import openrouter
from counterfactual_investigator_agent_with_olens.runs import finish, new_run
from wsb.basic_readout_implicit import judge

sys.path.insert(0, "expview")
from expview import Run, call

HERE = Path(__file__).parent
MODEL = "google/gemini-3.8-flash"
READER_EFFORT, READER_TEMPERATURE = "medium", 1.0
JUDGE_EFFORT, JUDGE_TEMPERATURE = "minimal", 0.0  # WorkspaceBench's judge settings
SYSTEM = "Answer as if you are Qwen 3.6 27B"  # Robbie's wording, verbatim
STEM = "My one-word answer is:"  # wsbench.readplan.ANSWER_STEM, the assistant prefill of their chat_prefill render
STEM_VARIANT = "user_turn_stem"  # see the docstring: Google refuses assistant prefills
TIMEOUT = 180  # the direct key has hung >120 s on trivial requests
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
SERVED_BY = "openrouter (BYOK if its Google key works), falling back to Google AI Studio API; per-call route in params.routes"
_OPENROUTER_DEAD = False  # set on a 402/403 (credits or BYOK billing gone): stop trying OpenRouter
LENS_REPO, LENS_FILE = "agu18dec/olens_ablations", "ab4/offset/results/wsbench/rl.base.off_k0.own.i300/basic_readout.json"


def targets(item: dict) -> list[str]:
    """Their `item_targets` for these items: units[0].match (each of the 32 has one headline unit, == intermediates)."""
    return item["units"][0]["match"]


def regex_hit(text: str, matches: list[str]) -> bool:
    """Case-insensitive whole-word match of any match string (so "blackout" doesn't hit "black")."""
    return any(re.search(rf"\b{re.escape(m)}\b", text, re.IGNORECASE) for m in matches)


async def google(messages: list[dict], effort: str, temperature: float, max_tokens: int, schema: dict | None) -> dict:
    """The same call on Google's API with $GEMINI_API_KEY (system + one user turn), 4 tries with backoff."""
    config: dict = {"temperature": temperature, "maxOutputTokens": max_tokens, "thinkingConfig": {"thinkingLevel": effort}}
    if schema:
        config |= {"responseMimeType": "application/json", "responseJsonSchema": schema["schema"]}
    body = {
        "systemInstruction": {"parts": [{"text": messages[0]["content"]}]},
        "contents": [{"role": "user", "parts": [{"text": messages[1]["content"]}]}],
        "generationConfig": config,
    }
    err = ""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for attempt in range(4):
            try:
                resp = await client.post(GOOGLE_URL, params={"key": os.environ["GEMINI_API_KEY"]}, json=body)
                if resp.status_code not in (429, 500, 503):
                    resp.raise_for_status()
                    out = resp.json()
                    text = "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"] if not p.get("thought"))
                    return {"text": text, "usage": out.get("usageMetadata"), "route": "google-direct", "key": "GEMINI_API_KEY"}
                err = f"{resp.status_code} {resp.text[:300]}"
            except httpx.TimeoutException:
                err = "timeout"
            await asyncio.sleep(10 * 2**attempt)
    raise RuntimeError(f"Google API failed 4 times: {err}")


async def gemini(messages: list[dict], effort: str, temperature: float, max_tokens: int, schema: dict | None = None) -> dict:
    """One Gemini 3.8 Flash call: OpenRouter first (BYOK when its Google key works), else Google directly."""
    global _OPENROUTER_DEAD
    if not _OPENROUTER_DEAD:
        params: dict = {}
        if schema:  # their judge request: strict schema, provider must honour it
            params = {
                "response_format": {"type": "json_schema", "json_schema": schema},
                "extra_body": {"provider": {"require_parameters": True}},
            }
        try:
            r = await openrouter.chat(
                MODEL, messages, effort=effort, temperature=temperature, max_tokens=max_tokens, timeout=TIMEOUT, attempts=3, **params
            )
            usage = r.usage.model_dump() if r.usage else None
            return {"text": r.choices[0].message.content or "", "usage": usage, "route": "openrouter", "key": "OPENROUTER_API_KEY"}
        except openai.APIStatusError as e:
            _OPENROUTER_DEAD = e.status_code in (402, 403)
            print(f"OpenRouter {e.status_code}; falling back to Google direct", flush=True)
        except (openai.APITimeoutError, openai.APIConnectionError, openrouter.NoChoices) as e:
            print(f"OpenRouter {type(e).__name__}; falling back to Google direct", flush=True)
    return await google(messages, effort, temperature, max_tokens, schema)


async def draw(item: dict, i: int, sem: asyncio.Semaphore) -> dict:
    """One reader continuation and its judge call."""
    user = f'{item["prompt"]}\n\nComplete this: "{STEM}"'
    reader_msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    async with sem:
        r = await gemini(reader_msgs, READER_EFFORT, READER_TEMPERATURE, 4000)
    answer = r["text"].strip()
    sample = answer or "(empty)"
    judge_msgs = [{"role": "system", "content": judge.SYSTEM}, {"role": "user", "content": judge.render_user(targets(item), [sample])}]
    async with sem:
        j = await gemini(judge_msgs, JUDGE_EFFORT, JUDGE_TEMPERATURE, 8000, judge.SCHEMA)
    raw = j["text"]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    v = judge.verdict(parsed, [sample])
    print(f"{item['name']:>24} s{i}  {answer[:40]!r:44} judge={v['expressed']!s:5} regex={regex_hit(answer, targets(item))}", flush=True)
    return {
        "item": item["name"],
        "draw": i,
        "targets": targets(item),
        "answer": answer,
        "judge": v,
        "regex_hit": regex_hit(answer, targets(item)),
        "reader": {"messages": reader_msgs, "output": answer, "usage": r["usage"], "route": r["route"], "key": r["key"]},
        "judge_call": {"messages": judge_msgs, "output": raw, "usage": j["usage"], "route": j["route"], "key": j["key"]},
    }


def lens_rows(items: list[dict]) -> dict[str, dict]:
    """The lens arm's per-item verdicts on our 32 items: pass = any judged layer expressed."""
    path = hf_hub_download(LENS_REPO, LENS_FILE)
    data = json.loads(Path(path).read_text())
    names = {it["name"] for it in items}
    rows = {}
    for row in data["rows"]:
        if row["id"] in names:
            layers = {int(k): bool(v["expressed"]) for k, v in row["layers"].items()}
            assert row["targets"] == targets(next(it for it in items if it["name"] == row["id"])), row["id"]
            rows[row["id"]] = {
                "pass": any(layers.values()),
                "their_pass": bool(row["pass"]),
                "earliest_layer": row["earliest_layer"],
                "n_layers_expressed": sum(layers.values()),
            }
    return {"config": data["config"], "n_items_total": data["n_items"], "overall_pass_rate": data["numbers"]["value"], "rows": rows}


def bootstrap_ci(per_item: list[float], n: int = 10_000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(per_item, k=len(per_item))) / len(per_item) for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n)]]


def summarize(items: list[dict], rows: list[dict], lens: dict) -> dict:
    by_item = {it["name"]: [r for r in rows if r["item"] == it["name"]] for it in items}
    table: list[dict[str, Any]] = []
    for it in items:
        rs = sorted(by_item[it["name"]], key=lambda r: r["draw"])
        lens_row = lens["rows"].get(it["name"])
        table.append(
            {
                "item": it["name"],
                "prompt": it["prompt"],
                "gold": targets(it),
                "answers": [r["answer"] for r in rs],
                "judge_pass": [r["judge"]["expressed"] for r in rs],
                "regex_hit": [r["regex_hit"] for r in rs],
                "lens_pass": lens_row["pass"] if lens_row else None,
                "lens_earliest_layer": lens_row["earliest_layer"] if lens_row else None,
            }
        )

    def rates(key: str) -> dict:
        p1 = [sum(t[key]) / len(t[key]) for t in table]
        p5 = [float(any(t[key])) for t in table]
        return {
            "pass@1": sum(p1) / len(p1),
            "pass@1_ci95": bootstrap_ci(p1),
            "pass@k": sum(p5) / len(p5),
            "pass@k_ci95": bootstrap_ci(p5),
            "items_missed": [t["item"] for t in table if not any(t[key])],
        }

    lens_pass = [t["lens_pass"] for t in table if t["lens_pass"] is not None]
    return {
        "n_items": len(table),
        "draws": len(table[0]["answers"]),
        "judge": rates("judge_pass"),
        "regex": rates("regex_hit"),
        "lens": {
            "arm": f"{LENS_REPO}:{LENS_FILE}",
            "judge_model": lens["config"]["judge_model"],
            "prompt_version": lens["config"]["prompt_version"],
            "pass": sum(lens_pass) / len(lens_pass),
            "n": len(lens_pass),
            "items_missed": [t["item"] for t in table if t["lens_pass"] is False],
        },
        "table": table,
    }


def markdown(summary: dict) -> str:
    j, g, lens = summary["judge"], summary["regex"], summary["lens"]
    k = summary["draws"]
    lines = [
        f"| metric | pass@1 (mean of {k} draws) | pass@{k} (any draw) |",
        "| --- | --- | --- |",
        (
            f"| judge (their prompt, Flash minimal) | {j['pass@1']:.2f} [{j['pass@1_ci95'][0]:.2f}, {j['pass@1_ci95'][1]:.2f}] | "
            f"{j['pass@k']:.2f} [{j['pass@k_ci95'][0]:.2f}, {j['pass@k_ci95'][1]:.2f}] |"
        ),
        (
            f"| regex (any match string, case-insensitive) | {g['pass@1']:.2f} [{g['pass@1_ci95'][0]:.2f}, {g['pass@1_ci95'][1]:.2f}] | "
            f"{g['pass@k']:.2f} [{g['pass@k_ci95'][0]:.2f}, {g['pass@k_ci95'][1]:.2f}] |"
        ),
        f"| lens, any of 11 layers ({lens['judge_model']}) | | {lens['pass']:.2f} (n={lens['n']}) |",
        "",
        "| item | gold | Gemini's answers | judge | regex | lens |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for t in summary["table"]:
        answers = ", ".join(a.removeprefix(STEM).strip().replace("|", "/").replace("\n", " ")[:30] for a in t["answers"])
        lines.append(
            f"| {t['item']} | {', '.join(t['gold'])} | {answers} | {sum(t['judge_pass'])}/{k} | {sum(t['regex_hit'])}/{k} | "
            f"{'pass' if t['lens_pass'] else 'miss' if t['lens_pass'] is False else '?'} |"
        )
    return "\n".join(lines)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="first N items only (smoke)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--slug", default="implicit-flash")
    ap.add_argument("--no-sync", action="store_true", help="finish without uploading to HF")
    args = ap.parse_args()

    bank = json.loads((HERE / "items.json").read_text())
    items = bank["items"][: args.limit or None]
    run = new_run(
        "wsb_basic_readout",
        args.slug,
        description="Gemini 3.8 Flash told to answer as Qwen3.6-27B guesses Qwen's one-word favourites on WorkspaceBench's 32 "
        "basic_readout implicit items; graded with WorkspaceBench's bank judge prompt (Flash, effort minimal) and a regex; "
        "compared with the lens arm rl.base.off_k0.own.i300 (Opus-judged RL ablation).",
        models=[
            {"name": MODEL, "role": "reader", "served_by": SERVED_BY, "effort": READER_EFFORT, "temperature": READER_TEMPERATURE},
            {"name": MODEL, "role": "judge", "served_by": SERVED_BY, "effort": JUDGE_EFFORT, "temperature": JUDGE_TEMPERATURE},
        ],
        params={
            "draws": args.draws,
            "n_items": len(items),
            "system": SYSTEM,
            "stem": STEM,
            "stem_variant": STEM_VARIANT,
            "user_template": '{prompt}\n\nComplete this: "{stem}"',
            "reader_max_tokens": 4000,
            "judge_prompt_version": judge.PROMPT_VERSION,
            "judge_max_tokens": 8000,
            "bank": bank["source"],
            "lens": {"repo": LENS_REPO, "file": LENS_FILE},
            "timeout_s": TIMEOUT,
        },
    )
    print("run:", run)
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(draw(it, i, sem) for it in items for i in range(args.draws)), return_exceptions=True)
    rows = [r for r in results if isinstance(r, dict)]
    failed = [repr(r) for r in results if not isinstance(r, dict)]
    print(f"{len(rows)} draws, {len(failed)} failed", *failed[:5], sep="\n")
    (run / "responses.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    lens = lens_rows(items)
    (run / "lens.json").write_text(json.dumps(lens, indent=1) + "\n")
    summary = summarize(items, rows, lens)
    calls = [(role, r[key]) for r in rows for role, key in (("reader", "reader"), ("judge", "judge_call"))]
    usage = [c["usage"] or {} for _, c in calls]
    summary["failed_draws"] = failed
    summary["spend"] = {
        "requests_succeeded": len(calls),  # retries and OpenRouter attempts before a fallback aren't counted
        "routes": dict(Counter(f"{role}: {c['route']} ({c['key']})" for role, c in calls)),
        "openrouter_cost_usd": sum(u.get("cost") or 0 for u in usage),
        "is_byok": sorted({str(u.get("is_byok")) for u in usage if "cost" in u}),
    }
    meta = json.loads((run / "meta.json").read_text())
    meta["params"]["routes"] = summary["spend"]["routes"]
    (run / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (run / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    (run / "summary.md").write_text(markdown(summary) + "\n")
    print(markdown(summary))
    print("spend:", summary["spend"])

    ev = Run(
        str(run / "expview"),
        title="WSB basic_readout implicit: Gemini 3.8 Flash as Qwen3.6-27B",
        description="One item per (bank item, draw). `judge_pass`: WorkspaceBench's bank judge says the target is expressed "
        "(quote verified); `regex_hit`: a match string appears in the answer; `lens_pass`: the lens arm passed this item.",
        label_order=["judge_pass", "regex_hit", "lens_pass"],
    )
    for r in rows:
        lens_row = lens["rows"].get(r["item"])
        ev.add(
            id=f"{r['item']}/s{r['draw']}",
            group=r["item"],
            labels={
                "judge_pass": r["judge"]["expressed"],
                "regex_hit": r["regex_hit"],
                "lens_pass": lens_row["pass"] if lens_row else None,
            },
            calls=[
                call(
                    "reader (Gemini 3.8 Flash as Qwen)",
                    MODEL,
                    messages=r["reader"]["messages"],
                    output={"role": "assistant", "content": r["answer"]},
                    params={"effort": READER_EFFORT, "temperature": READER_TEMPERATURE, "route": r["reader"]["route"]},
                ),
                call(
                    "judge (WSB bank judge)",
                    MODEL,
                    messages=r["judge_call"]["messages"],
                    output={"role": "assistant", "content": r["judge_call"]["output"]},
                    params={"effort": JUDGE_EFFORT, "temperature": JUDGE_TEMPERATURE, "route": r["judge_call"]["route"]},
                    parsed=r["judge"],
                ),
            ],
            extra={"gold": r["targets"], "answer": r["answer"]},
        )
    finish(run, sync=not args.no_sync)
    print("finished", run)


if __name__ == "__main__":
    asyncio.run(main())
