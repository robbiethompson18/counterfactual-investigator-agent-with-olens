# ruff: noqa: E501
"""Judge: for one investigation and one candidate edit (real or null, role hidden), did the investigator name the
feature the edit changes? Same question as baseline_hyp/judge.py. Claude Haiku 5.5 (Kylie asked for cheaper than Sonnet 5.5), 1 replicate.

  ~/workspace/donehack/.venv/bin/python envguess/judge.py --inv data/envguess/inv/v1 --out data/envguess/judge/v1
"""

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
SETTINGS = REPO / "data" / "envguess" / "settings"
MODEL = "anthropic/claude-haiku-5.5"  # Kylie 2026-10-26: cheaper than Sonnet 5.5 for this judge

PROMPT = """An investigator studied why a language model shows a behaviour in some setup, and submitted the hypotheses below. Separately, someone made a specific change to the setup's inputs, shown as a diff (a/ is the original setup, b/ the changed one).

Your job: decide whether the investigator's output NAMED the feature of the setup that this diff changes, as something that causes, enables or modulates the behaviour. It counts if a hypothesis says this feature matters, or if a proposed test changes this same feature (in either direction, with any wording). It does not count if the investigator only mentions the feature in passing without tying it to the behaviour, or names a different feature. You are not judging whether the hypothesis is correct.

<behaviour>
{behaviour}
</behaviour>

<investigator_output>
{output}
</investigator_output>

<diff>
{diff}
</diff>

Think briefly, then end with exactly one line of JSON: {{"named": true or false, "which": [the hypothesis numbers that name it, or empty]}}"""


def format_output(res):
    lines = [f"Summary: {res.get('summary', '')}"]
    for i, h in enumerate(res.get("hypotheses") or [], 1):
        lines.append(
            f"\nHypothesis {i} (confidence {h.get('confidence')}): {h.get('hypothesis')}\n  Evidence: {h.get('evidence')}\n  Proposed test: {h.get('test')}"
        )
    return "\n".join(lines)


def parse(content):
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", content or ""))):
        try:
            j = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(j.get("named"), bool):
            w = j.get("which") or []
            w = w if isinstance(w, list) else [int(x) for x in re.findall(r"\d+", str(w))]
            return {"named": j["named"], "which": [int(x) for x in w if str(x).isdigit()]}
    return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", type=int, default=48)
    a = ap.parse_args()
    key = Path("/secrets/openrouter").read_text().strip()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for f in sorted(Path(a.inv).glob("*.json")):
        rec = json.loads(f.read_text())["record"]
        if not rec.get("result"):
            continue
        sid, cid = rec["view"].rsplit("__", 1)
        s = json.loads((SETTINGS / sid / "setting.json").read_text())
        for lab in s["edits"]:
            p = out / f"{f.stem}__{lab}.json"
            if not p.exists():
                diff = (SETTINGS / sid / "cases" / cid / "edits" / f"{lab}.diff").read_text()
                jobs.append((p, rec, sid, lab, PROMPT.format(behaviour=s["behaviour"], output=format_output(rec["result"]), diff=diff)))
    random.Random(0).shuffle(jobs)
    print(len(jobs), "judgements", flush=True)
    sem = asyncio.Semaphore(a.conc)
    total, done = 0.0, 0
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=a.conc + 10)) as client:

        async def one(p, rec, sid, lab, prompt):
            nonlocal total, done
            async with sem:
                body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4000, "temperature": 0}
                err, j = None, None
                for attempt in range(6):
                    try:
                        r = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            json=body,
                            headers={"Authorization": f"Bearer {key}"},
                            timeout=300,
                        )
                        j = r.json()
                        if "choices" in j and parse(j["choices"][0]["message"].get("content")):
                            break
                        err, j = j, None
                    except Exception as e:  # noqa: BLE001
                        err = repr(e)
                    await asyncio.sleep(2**attempt + random.random())
                if j is None:
                    print("FAILED", p.name, str(err)[:300], flush=True)
                    return
                content = j["choices"][0]["message"].get("content")
                cost = (j.get("usage") or {}).get("cost", 0) or 0
                total += cost
                done += 1
                p.write_text(
                    json.dumps(
                        {
                            "arm": rec["arm"],
                            "view": rec["view"],
                            "seed": rec["seed"],
                            "setting": sid,
                            "label": lab,
                            "model": MODEL,
                            "prompt": prompt,
                            "response": content,
                            "verdict": parse(content),
                            "cost": cost,
                            "usage": j.get("usage"),
                        },
                        indent=1,
                    )
                )
                if done % 50 == 0:
                    print(f"{done} done, ${total:.2f}", flush=True)

        await asyncio.gather(*[one(*x) for x in jobs])
    print(f"finished {done}, ${total:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
