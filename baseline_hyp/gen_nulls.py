"""Generate candidate null edits for one baseline variant: small changes to environment/ that a careful experimenter
would expect NOT to change the behaviour (irrelevant sentence added, meaning-preserving rewording, irrelevant
name/date/number changed). Each edit is one exact substring replacement in one environment file.
Whether they really are null is measured afterwards (nulls_run.py). The generator never sees hypotheses or known effects.

python gen_nulls.py --views a,b --k 5 --out nulls/edits.json
"""
# ruff: noqa: E501 - prompt/label text below is kept byte-identical to what the runs used

import argparse
import asyncio
import json
import re
from pathlib import Path

import agent
import httpx

PROMPT = """You are helping design a control condition for a behavioural experiment on a language model.

The model was run many times on the inputs below. The behaviour being measured: {behaviour}

Propose {k} small edits to these inputs that a careful experimenter would expect to have NO effect on how often this behaviour happens. Make them varied, for example:
- add one irrelevant, mundane sentence somewhere natural (office logistics, an unrelated fact),
- reword one sentence without changing its meaning or tone,
- change an irrelevant detail (a name, date, number or ID that the behaviour does not depend on),
- small formatting changes.
Each edit must change exactly one thing, must not touch anything that plausibly bears on the behaviour, and must keep the scenario coherent.

Each edit is one exact substring replacement in one file: "old" must appear exactly once in that file (copy it verbatim, including punctuation), "new" is its replacement.

<inputs>
{files}
</inputs>

Reply with a JSON array only, no prose: [{{"file": "<path as given>", "old": "...", "new": "...", "kind": "add_sentence|reword|detail|format", "why_null": "..."}}, ...]"""


def env_files(view):
    ed = Path("views") / view / "environment"
    return {str(p.relative_to(ed)): p.read_text(errors="replace") for p in sorted(ed.rglob("*")) if p.is_file()}


async def gen(client, key, view, behaviour, k, model):
    files = env_files(view)
    shown = "\n\n".join(f'<file path="{p}">\n{c}\n</file>' for p, c in files.items() if len(c) < 30000 and "base64" not in c)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(behaviour=behaviour, k=k, files=shown)}],
        "max_tokens": 16000,
        "reasoning": {"effort": "medium"},
        "provider": {"order": ["DeepInfra"], "allow_fallbacks": True},
    }
    for _ in range(3):
        j = await agent.chat(client, key, body)
        txt = j["choices"][0]["message"].get("content") or ""
        m = re.search(r"\[.*\]", txt, re.DOTALL)
        try:
            edits = json.loads(m.group(0))
        except (AttributeError, json.JSONDecodeError):
            continue
        ok = [
            e
            for e in edits
            if e.get("file") in files and files[e["file"]].count(e.get("old", "")) == 1 and e["old"] and e["old"] != e.get("new")
        ]
        if len(ok) >= k:
            return ok[:k]
    return ok


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="nulls/edits.json")
    ap.add_argument("--model", default="deepseek/deepseek-v4.1-flash-1017")
    a = ap.parse_args()
    key = agent.openrouter_key()
    idx = {e["env"]: e for e in json.loads(Path("index.json").read_text())}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    async with httpx.AsyncClient() as c:
        views = [v for v in a.views.split(",") if v not in res]
        got = await asyncio.gather(*[gen(c, key, v, idx[v.split("__")[0]]["behaviour"], a.k, a.model) for v in views])
    for v, g in zip(views, got):
        res[v] = [dict(e, arm=f"null{i + 1}") for i, e in enumerate(g)]
        print(v, len(g))
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
