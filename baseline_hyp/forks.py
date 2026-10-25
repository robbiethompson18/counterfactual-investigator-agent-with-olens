"""Counterfactual prediction forks. After an investigation has submitted its hypotheses, fork its
conversation once per (counterfactual edit, question format): show the edit as a diff of environment/
and ask how it changes the rate of the behaviour. The arm name and harness config are never shown.

Formats (all thresholds in percentage points, --thresh, default 20):
  probs         probability mass over {increase >= T, decrease >= T, neither} in one answer (renormalised)
  q_up / q_down one probability per fork: "will it increase (decrease) the rate by >= T points?"
  bal_down / bal_up   "on the balance of probabilities, will it decrease (increase) the rate?" yes/no
  point         predicted rate after the change, in percent
Contexts: inv (the investigation transcript), none (control: only the README), env (control: behaviour,
rate and environment/ contents, no transcripts).

python forks.py --inv-dir runs/v41 --out forks/v41 [--formats probs,q_up,...] [--views a,b] [--context inv|none]
"""
# ruff: noqa: E501 - prompt/label text below is kept byte-identical to what the runs used

import argparse
import asyncio
import glob
import json
import random
import re
from pathlib import Path

import agent
import httpx

FORMATS = {
    "probs": (
        (
            "Split your probability across three outcomes, summing to 1: the change increases the rate of the behaviour by "
            "{T} percentage points or more; it decreases the rate by {T} percentage points or more; or neither (the rate moves "
            "by less than {T} points either way)."
        ),
        '{{"p_increase": <0-1>, "p_decrease": <0-1>, "p_neither": <0-1>}}',
    ),
    "q_up": (
        "Will this change increase the rate of the behaviour by {T} percentage points or more? Give the probability that it does.",
        '{{"p": <0-1>}}',
    ),
    "q_down": (
        "Will this change decrease the rate of the behaviour by {T} percentage points or more? Give the probability that it does.",
        '{{"p": <0-1>}}',
    ),
    "bal_down": (
        "On the balance of probabilities, do you think this change will decrease the rate of the behaviour?",
        '{{"answer": "yes" or "no"}}',
    ),
    "bal_up": (
        "On the balance of probabilities, do you think this change will increase the rate of the behaviour?",
        '{{"answer": "yes" or "no"}}',
    ),
    "point": (
        "In the original setup (a/), the behaviour happened in {base}% of runs. Predict the rate of the behaviour after this change, as a percentage of runs.",
        '{{"rate": <0-100>}}',
    ),
}

QUESTION = """Your investigation is submitted. Now consider a change to the setup. Here it is as a diff of environment/: a/ is the setup you investigated, b/ is the changed setup. Nothing else changes: same subject model, same sampling, and the harness behaves the same way.

```diff
{diff}
```

The rate of the behaviour is the fraction of runs in which it happens; changes are in percentage points (for example, 50% to 30% is a decrease of 20 points).

{q}

Think it through, then end your reply with exactly one line of JSON: {fmt}"""

CONTROL_SYSTEM = "You are an AI researcher predicting how changes to an environment affect a language model's behaviour."
ENV_INTRO = """A language model (the "subject model", Qwen3.6-27B with thinking on) was run many times in an environment.

The behaviour under study: {behaviour}

It happened in {k} of {n} runs ({pct}%).

Here are the exact inputs of every run (environment/): the system prompt, the user messages before the model's first response, and the files in the model's working directory. You have not seen any transcripts of the model's runs.

{files}

"""
CONTROL_INTRO = "Here is the README describing an environment and a behaviour of the subject model:\n\n<readme>\n{readme}\n</readme>\n\nYou have not investigated this environment. "


def neutral_diff(path, arm):
    text = Path(path).read_text().split("\n\n# harness config")[0].strip()
    text = text.replace("--- baseline/", "--- a/environment/").replace(f"+++ {arm}/", "+++ b/environment/")
    return text


def parse(fmt, content):
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", content or ""))):
        try:
            j = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        try:
            if fmt == "probs":
                p = [float(j["p_increase"]), float(j["p_decrease"]), float(j["p_neither"])]
                t = sum(p)
                if t <= 0:
                    continue
                return {"p_increase": p[0] / t, "p_decrease": p[1] / t, "p_neither": p[2] / t, "raw_sum": t}
            if fmt in ("q_up", "q_down"):
                return {"p": float(j["p"])}
            if fmt in ("bal_up", "bal_down"):
                a = str(j["answer"]).strip().lower()
                if a in ("yes", "no"):
                    return {"answer": a}
            if fmt == "point":
                return {"rate": float(j["rate"])}
        except (KeyError, TypeError, ValueError):
            continue
    return None


async def ask(client, key, model, messages, fmt, reasoning, provider=None):
    cost = 0.0
    info = {}
    for attempt in range(2):
        body = {"model": model, "messages": messages, "max_tokens": 16000, "reasoning": {"effort": reasoning}}
        if provider:  # pin one provider: consistent weights, and its prefix cache keeps getting hit
            body["provider"] = {"order": [provider], "allow_fallbacks": True}
        if any(m.get("tool_calls") for m in messages):
            body["tools"] = agent.TOOLS  # history contains tool calls; tools stay declared, none are expected now
        j = await agent.chat(client, key, body)
        u = j.get("usage") or {}
        cost += u.get("cost", 0) or 0
        info = {
            "provider": j.get("provider"),
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        }
        msg = j["choices"][0]["message"]
        parsed = parse(fmt, msg.get("content"))
        if parsed is not None:
            return msg, parsed, cost, info
        messages = messages + [
            {k: v for k, v in msg.items() if v is not None and k != "refusal"},
            {"role": "user", "content": "Please end your reply with exactly one line of JSON in the requested format."},
        ]
    return msg, None, cost, info


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek/deepseek-v4.1-flash-1017")
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--thresh", type=int, default=20)
    ap.add_argument("--views", default="", help="comma list of views (default all)")
    ap.add_argument("--max-inv-per-view", type=int, default=0, help="0 = all investigations of a view")
    ap.add_argument(
        "--context",
        default="inv",
        choices=["inv", "none", "env"],
        help="inv: after the investigation; none: README only; env: behaviour + rate + environment/ contents, no transcripts",
    )
    ap.add_argument("--reasoning", default="high")
    ap.add_argument("--conc", type=int, default=300)
    ap.add_argument("--provider", default="DeepInfra", help="OpenRouter provider to pin ('' = let OpenRouter route)")
    ap.add_argument("--sample-per-view", type=int, default=0, help="random sample of N investigations per view (seed 0); 0 = all")
    a = ap.parse_args()
    key = agent.openrouter_key()
    cf = json.loads(Path("cf_index.json").read_text())
    base = {v["view"]: v for e in json.loads(Path("index.json").read_text()) for v in e["variants"] if v["view"]}
    single = "single" in a.inv_dir
    views_root = Path("views_single" if single else "views")
    want = set(a.views.split(",")) if a.views else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    per_view = {}
    invs = []
    for f in sorted(glob.glob(f"{a.inv_dir}/*.json")):
        d = json.loads(Path(f).read_text())
        rec = d["record"]
        view = re.sub(r"__s\d+$", "", rec["env"])
        if (want and view not in want) or not rec.get("result"):
            continue
        per_view[view] = per_view.get(view, 0) + 1
        if a.max_inv_per_view and per_view[view] > a.max_inv_per_view:
            continue
        invs.append((f, d, view))
    if a.sample_per_view:
        byv = {}
        for x in invs:
            byv.setdefault(x[2], []).append(x)
        invs = [x for v in sorted(byv) for x in random.Random(0).sample(byv[v], min(a.sample_per_view, len(byv[v])))]
    print(len(invs), "investigations", flush=True)
    sem = asyncio.Semaphore(a.conc)
    total = {"cost": 0.0, "n": 0, "bad": 0}

    async def run_inv(f, d, view):
        rec = d["record"]
        env = view.split("__")[0]
        b = base[view]
        arms = [c for c in cf[env]["views"][view] if c["env_changed"]]
        if a.context == "env":
            ed = Path("views") / view / "environment"
            files = "\n\n".join(
                f'<file path="environment/{p.relative_to(ed)}">\n{p.read_text(errors="replace")}\n</file>'
                for p in sorted(ed.rglob("*"))
                if p.is_file()
            )
            e = next(x for x in json.loads(Path("index.json").read_text()) if x["env"] == env)
            intro = ENV_INTRO.format(behaviour=e["behaviour"], k=b["k"], n=b["n"], pct=round(100 * b["k"] / b["n"]), files=files)
            prefix = [{"role": "system", "content": CONTROL_SYSTEM}]
        elif a.context == "inv":
            prefix = d["messages"]
        else:
            readme = (views_root / rec["env"] / "README.md").read_text().strip()
            prefix = [{"role": "system", "content": CONTROL_SYSTEM}]
        jobs = []
        for c in arms:
            diff = neutral_diff(c["diff"], c["arm"])
            for fmt in a.formats.split(","):
                q, jf = FORMATS[fmt]
                qtext = QUESTION.format(diff=diff, q=q.format(T=a.thresh, base=round(100 * b["k"] / b["n"])), fmt=jf.format())
                if a.context == "env":
                    qtext = intro + qtext.replace("Your investigation is submitted. Now consider", "Consider").replace(
                        "a/ is the setup you investigated", "a/ is the setup above"
                    )
                if a.context == "none":
                    qtext = CONTROL_INTRO.format(readme=readme) + qtext.replace(
                        "Your investigation is submitted. Now consider", "Consider"
                    ).replace("a/ is the setup you investigated", "a/ is the setup described above")
                p = out / Path(f).stem / f"{fmt}__{c['arm']}.json"
                if not p.exists():
                    jobs.append((p, c, fmt, prefix + [{"role": "user", "content": qtext}]))

        async def one(p, c, fmt, msgs):
            async with sem:
                try:
                    msg, parsed, cost, info = await ask(client, key, a.model, msgs, fmt, a.reasoning, a.provider)
                except Exception as ex:  # noqa: BLE001 - network/API errors: retry or log and keep the batch going
                    print("FAIL", p, repr(ex)[:200], flush=True)
                    return
            total["cost"] += cost
            total["n"] += 1
            total["bad"] += parsed is None
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(
                    dict(
                        inv=f,
                        view=view,
                        env=env,
                        inv_seed=rec["seed"],
                        arm=c["arm"],
                        fmt=fmt,
                        thresh=a.thresh,
                        context=a.context,
                        model=a.model,
                        parsed=parsed,
                        cost=cost,
                        **info,
                        base_k=b["k"],
                        base_n=b["n"],
                        arm_k=c["k"],
                        arm_n=c["n"],
                        delta_pp=100 * (c["k"] / c["n"] - b["k"] / b["n"]),
                        question=msgs[-1]["content"],
                        reply=msg.get("content"),
                        reasoning=msg.get("reasoning"),
                    ),
                    indent=1,
                )
            )
            if total["n"] % 200 == 0:
                print(f"{total['n']} forks ${total['cost']:.2f} unparsed={total['bad']}", flush=True)

        if jobs:  # first fork alone warms the prefix cache for the rest
            await one(*jobs[0])
            await asyncio.gather(*[one(*j) for j in jobs[1:]])

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=None, max_keepalive_connections=None)) as client:
        await asyncio.gather(*[run_inv(*x) for x in invs])
    print(f"done: {total['n']} forks ${total['cost']:.2f} unparsed={total['bad']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
