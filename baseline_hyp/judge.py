""" "Hypothesis generation" metric: an LLM judge reads one investigation's submitted hypotheses and one
counterfactual edit (diff of environment/, arm name hidden), and says whether the investigator named
the feature that the edit changes as something that drives the behaviour.

python judge.py --inv-dir runs/v41 --out judge/v41 [--model ...]
"""
# ruff: noqa: E501 - prompt/label text below is kept byte-identical to what the runs used

import argparse
import asyncio
import glob
import json
import re
from pathlib import Path

import agent
import httpx
from forks import neutral_diff

PROMPT = """An investigator studied why a language model shows a behaviour in some environment, and submitted the hypotheses below. Separately, someone made a specific change to the environment's inputs, shown as a diff (a/ is the original setup, b/ the changed one).

Your job: decide whether the investigator's output NAMED the feature of the setup that this diff changes, as something that causes, enables or modulates the behaviour. It counts if a hypothesis says this feature matters, or if a proposed test changes this same feature (in either direction, with any wording). It does not count if the investigator only mentions the feature in passing without tying it to the behaviour, or names a different feature.

<behaviour>
{behaviour}
</behaviour>

<investigator_output>
{output}
</investigator_output>

<diff>
{diff}
</diff>

Think briefly, then end with exactly one line of JSON: {{"named": true or false, "which": "<the hypothesis number(s) that name it, or empty>"}}"""


def format_output(res):
    lines = [f"Summary: {res.get('summary', '')}"]
    for i, h in enumerate(res.get("hypotheses") or [], 1):
        lines.append(f"\nHypothesis {i}: {h.get('hypothesis')}\n  Evidence: {h.get('evidence')}\n  Proposed test: {h.get('test')}")
    return "\n".join(lines)


def parse(content):
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", content or ""))):
        try:
            j = json.loads(m.group(0))
            if isinstance(j.get("named"), bool):
                return {"named": j["named"], "which": str(j.get("which", ""))}
        except json.JSONDecodeError:
            continue
    return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek/deepseek-v4.1-flash-1017")
    ap.add_argument("--conc", type=int, default=300)
    ap.add_argument("--provider", default="DeepInfra")
    a = ap.parse_args()
    key = agent.openrouter_key()
    cf = json.loads(Path("cf_index.json").read_text())
    idx = {e["env"]: e for e in json.loads(Path("index.json").read_text())}
    base = {v["view"]: v for e in idx.values() for v in e["variants"] if v["view"]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for f in sorted(glob.glob(f"{a.inv_dir}/*.json")):
        rec = json.loads(Path(f).read_text())["record"]
        if not rec.get("result"):
            continue
        view = re.sub(r"__s\d+$", "", rec["env"])
        env = view.split("__")[0]
        if view not in base:  # pooled-variant runs have no single baseline variant
            continue
        output = format_output(rec["result"])
        for c in cf[env]["views"][view]:
            if not c["env_changed"]:
                continue
            p = out / Path(f).stem / f"{c['arm']}.json"
            if not p.exists():
                jobs.append(
                    (
                        p,
                        f,
                        rec,
                        view,
                        env,
                        c,
                        PROMPT.format(behaviour=idx[env]["behaviour"], output=output, diff=neutral_diff(c["diff"], c["arm"])),
                    )
                )
    print(len(jobs), "judgements to run", flush=True)
    sem = asyncio.Semaphore(a.conc)
    tot = {"n": 0, "cost": 0.0, "bad": 0}
    b = base

    async def one(p, f, rec, view, env, c, prompt):
        async with sem:
            parsed, cost = None, 0.0
            msgs = [{"role": "user", "content": prompt}]
            try:
                for _ in range(2):
                    body = {"model": a.model, "messages": msgs, "max_tokens": 8000, "reasoning": {"effort": "low"}}
                    if a.provider:
                        body["provider"] = {"order": [a.provider], "allow_fallbacks": True}
                    j = await agent.chat(client, key, body)
                    cost += (j.get("usage") or {}).get("cost", 0) or 0
                    msg = j["choices"][0]["message"]
                    parsed = parse(msg.get("content"))
                    if parsed:
                        break
            except Exception as ex:  # noqa: BLE001 - network/API errors: retry or log and keep the batch going
                print("FAIL", p, repr(ex)[:200], flush=True)
                return
        tot["n"] += 1
        tot["cost"] += cost
        tot["bad"] += parsed is None
        p.parent.mkdir(parents=True, exist_ok=True)
        bv = b[view]
        p.write_text(
            json.dumps(
                {
                    "inv": f,
                    "view": view,
                    "env": env,
                    "arm": c["arm"],
                    "model": a.model,
                    "parsed": parsed,
                    "cost": cost,
                    "base_k": bv["k"],
                    "base_n": bv["n"],
                    "arm_k": c["k"],
                    "arm_n": c["n"],
                    "delta_pp": 100 * (c["k"] / c["n"] - bv["k"] / bv["n"]),
                    "reply": msg.get("content"),
                },
                indent=1,
            )
        )
        if tot["n"] % 500 == 0:
            print(f"{tot['n']} ${tot['cost']:.2f} unparsed={tot['bad']}", flush=True)

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=None, max_keepalive_connections=None)) as client:
        await asyncio.gather(*[one(*j) for j in jobs])
    print(f"done {tot['n']} ${tot['cost']:.2f} unparsed={tot['bad']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
