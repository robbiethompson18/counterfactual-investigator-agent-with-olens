"""Rate auditor: an LLM spot-checks ~10 graded transcripts per arm against the behaviour sentence and the grader,
and scores the node 1 / 2 / 3 (prompts/auditor.md). Saved as <node>/audit/<batch>.json; the score goes in node.json."""

import json
import random
import re
from pathlib import Path

from . import llm
from .measure import load
from .node import ARMS, Node

PROMPTS = Path(__file__).parent / "prompts"
CUT = {"prompt": 1500, "reasoning": 1200, "text": 3000, "final": 8000, "args": 1500, "output": 1500, "file": 1500}


def _cut(s, n):
    s = s if isinstance(s, str) else json.dumps(s)
    return s if len(s) <= n else s[: n // 2] + f"\n...[{len(s) - n} chars cut]...\n" + s[-n // 2 :]


def sample(rows, k, seed):
    """Up to k scorable rows, half of each verdict where possible, plus up to 3 unscorable ones."""
    rng = random.Random(seed)
    yes, no, none = ([r for r in rows if r["verdict"] is v] for v in (True, False, None))
    for xs in (yes, no, none):
        rng.shuffle(xs)
    n_yes = min(len(yes), max(k // 2, k - len(no)))
    pick = yes[:n_yes] + no[: k - n_yes] + none[:3]
    return sorted(pick, key=lambda r: r["sample"])


def render(arm_files: dict, r: dict, rid: str) -> str:
    rec = r["record"]
    out = [
        f"### {rid} (verdict: {r['verdict']}{', grader error: ' + r['grader_error'][:200] if r.get('grader_error') else ''})",
        f"finish: {rec.get('finish')}; turns: {rec.get('n_turns')}"
        + (f"; read-only files tampered: {rec['readonly_tampered']}" if rec.get("readonly_tampered") else ""),
    ]
    turns = rec.get("turns", [])
    for i, t in enumerate(turns):
        last = i == len(turns) - 1
        if t.get("reasoning"):
            out.append(f"[thinking] {_cut(t['reasoning'], CUT['reasoning'])}")
        if t.get("text"):
            out.append(f"[assistant] {_cut(t['text'], CUT['final'] if last else CUT['text'])}")
        for c in t.get("tool_calls", []):
            out.append(f"[tool call {c['name']}] {_cut(c['args'], CUT['args'])}\n[tool output] {_cut(c.get('output', ''), CUT['output'])}")
    changed = {p: c for p, c in rec.get("files_after", {}).items() if arm_files.get(p) != c}
    if changed:
        out.append("[files changed or created by the run]\n" + "\n".join(f"--- {p}\n{_cut(c, CUT['file'])}" for p, c in changed.items()))
    return "\n".join(out)


def parse(text):
    """The last JSON object starting with {"score" in the reply (it may span lines)."""
    text = text or ""
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r'\{\s*"score"', text))):
        try:
            obj, _ = dec.raw_decode(text, m.start())
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    return None


async def run_audit(node: Node, batch: str, k: int = 10, model=llm.DS) -> dict:
    arms = {a: node.arm(a) for a in ARMS}
    rows = {a: load(node, batch, a) for a in ARMS}
    kn = {a: (sum(r["verdict"] is True for r in rows[a]), sum(r["verdict"] in (True, False) for r in rows[a])) for a in ARMS}
    un = {a: len(rows[a]) - kn[a][1] for a in ARMS}
    parts = []
    for a in ARMS:
        prompt = "\n\n".join(m["content"] for m in arms[a].messages if m["role"] in ("system", "user"))
        parts.append(f"## {a.upper()} arm\nPrompt (cut): {_cut(prompt, CUT['prompt'])}\n")
        parts += [render(arms[a].files, r, f"{a}-{r['sample']:02d}") for r in sample(rows[a], k, f"{node.id}-{batch}-{a}")]
    prompt = (PROMPTS / "auditor.md").read_text()
    for key, val in {
        "behaviour": node.behaviour,
        "grader": node.grader_src,
        "kb": kn["baseline"][0],
        "nb": kn["baseline"][1],
        "kr": kn["real"][0],
        "nr": kn["real"][1],
        "kn": kn["null"][0],
        "nn": kn["null"][1],
        "ub": un["baseline"],
        "ur": un["real"],
        "un": un["null"],
        "transcripts": "\n\n".join(parts),
    }.items():
        prompt = prompt.replace("{" + key + "}", str(val))
    msg, usage = await llm.ask([{"role": "user", "content": prompt}], model=model)
    parsed = parse(msg["content"])
    res = {
        "batch": batch,
        "model": model,
        "score": (parsed or {}).get("score"),
        "parsed": parsed,
        "reply": msg["content"],
        "reasoning": msg.get("reasoning"),
        "prompt": prompt,
        "cost": usage["cost"],
    }
    (node.dir / "audit").mkdir(exist_ok=True)
    (node.dir / "audit" / f"{batch}.json").write_text(json.dumps(res, indent=1) + "\n")
    node.update(audit_score=res["score"])
    return res
