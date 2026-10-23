"""Innocuousness rating: how hard it is to tell that a node's REAL and NULL edits could move the behaviour.

A judge (DSv4.1 Flash with bash in a podman Box) compares two nodes A and B. Phase 1: it reads both baselines,
mounted read-only at /ro/A/baseline and /ro/B/baseline (every arm file except meta.json; no runs, scores, picker or
audit data, no hypothesis). Phase 2: it is shown each node's behaviour sentence and REAL / NULL diffs, may look at
the context of the edits, and calls verdict(winner="A"|"B", reason). Prompts: prompts/innocuous*.md.

    data/<run>/innocuous.jsonl          one comparison per line (append-only): {id, t, a, b, winner, reason, ...}
    data/<run>/innocuous/<id>.json      the comparison's full input and transcript
    data/<run>/innocuous_ratings.json   {node: {rating, n, wins}}, rewritten after every comparison

Ratings are a Bradley-Terry fit over all comparisons with a N(0, PRIOR_SD^2) prior on each node's log-strength,
reported on the Elo scale (400 / ln 10 per unit, centred on 1500).
"""

import asyncio
import json
import logging
import math
import os
import random
import shutil
import statistics
import tempfile
import time
import uuid
from pathlib import Path

from . import llm
from .agent import _clean
from .node import Node, arm_text, unified_diff
from .sandbox import Box

log = logging.getLogger(__name__)
PROMPTS = Path(__file__).resolve().parent / "prompts"
HIDDEN = ("meta.json",)
K = 4  # comparisons per new node
EXPLORE_TURNS = 8
VERDICT_TURNS = 4
MAX_DIFF = 8000
MAX_OUT = 8000
PRIOR_SD = 1.0  # log-odds units (~175 Elo)
ELO = 400 / math.log(10)
BASE = 1500.0

BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command (read-only view of /ro/A and /ro/B). 60 s timeout; long output is truncated.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}
VERDICT = {
    "type": "function",
    "function": {
        "name": "verdict",
        "description": "Your answer: which environment's edits are more innocuous.",
        "parameters": {
            "type": "object",
            "properties": {"winner": {"type": "string", "enum": ["A", "B"]}, "reason": {"type": "string"}},
            "required": ["winner", "reason"],
        },
    },
}


def run_dir(node: Node) -> Path:
    return node.dir.parent.parent  # data/<run>/nodes/<id>


def comparisons(run: Path) -> list[dict]:
    p = Path(run) / "innocuous.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def fit(rows: list[dict], iters=200) -> dict[str, dict]:
    """Bradley-Terry MAP by coordinate Newton steps: {node: {rating (Elo scale), n, wins}}."""
    games = [(r["winner"], r["b"] if r["winner"] == r["a"] else r["a"]) for r in rows if r.get("winner") in (r.get("a"), r.get("b"))]
    ids = sorted({x for g in games for x in g})
    th = dict.fromkeys(ids, 0.0)
    opp = {i: [] for i in ids}  # (other, 1 if i won else 0)
    for w, l in games:
        opp[w].append((l, 1))
        opp[l].append((w, 0))
    for _ in range(iters):
        delta = 0.0
        for i in ids:
            g, h = -th[i] / PRIOR_SD**2, 1 / PRIOR_SD**2
            for j, y in opp[i]:
                p = 1 / (1 + math.exp(th[j] - th[i]))
                g += y - p
                h += p * (1 - p)
            th[i] += g / h
            delta = max(delta, abs(g / h))
        if delta < 1e-6:
            break
    return {i: {"rating": round(BASE + ELO * th[i], 1), "n": len(opp[i]), "wins": sum(y for _, y in opp[i])} for i in ids}


def ratings(store_or_run) -> dict[str, float]:
    """{node_id: rating} from every comparison in the run (refit on demand)."""
    run = getattr(store_or_run, "dir", store_or_run)
    return {k: v["rating"] for k, v in fit(comparisons(run)).items()}


def write_ratings(run: Path) -> dict:
    run = Path(run)
    r = fit(comparisons(run))
    tmp = run / f"innocuous_ratings.json.{os.getpid()}.{uuid.uuid4().hex[:6]}"
    tmp.write_text(json.dumps(r, indent=1, sort_keys=True) + "\n")
    tmp.replace(run / "innocuous_ratings.json")
    return r


def saved_ratings(run: Path) -> dict[str, float]:
    """Ratings as last written (cheap; what the Pool reads)."""
    p = Path(run) / "innocuous_ratings.json"
    if not p.exists():
        return {}
    try:
        return {k: v["rating"] for k, v in json.loads(p.read_text()).items()}
    except (OSError, ValueError, KeyError):
        return {}


def values(nodes: list[Node]) -> dict[str, float]:
    """Each node's rating; nodes not compared yet get the median of the rated ones (BASE if none is rated)."""
    if not nodes:
        return {}
    saved = saved_ratings(run_dir(nodes[0]))
    rated = [saved[n.id] for n in nodes if n.id in saved] or list(saved.values())
    med = statistics.median(rated) if rated else BASE
    return {n.id: saved.get(n.id, med) for n in nodes}


def _clip(s: str, n=MAX_DIFF) -> str:
    return s if len(s) <= n else s[: n // 2] + f"\n...[{len(s) - n} chars cut]...\n" + s[-n // 2 :]


def _diffs(node: Node) -> tuple[str, str]:
    d = node.dir
    return tuple(_clip(unified_diff(d / "baseline", d / a, HIDDEN)) or "(no change)" for a in ("real", "null"))


def _stage(a: Node, b: Node) -> Path:
    """Temp tree with A/baseline and B/baseline (meta.json left out), world-readable for the container user."""
    root = Path(tempfile.mkdtemp(prefix="evolve_innoc_"))
    for tag, n in (("A", a), ("B", b)):
        for rel, text in arm_text(n.dir / "baseline", HIDDEN).items():
            p = root / tag / "baseline" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    for p in [root, *root.rglob("*")]:
        os.chmod(p, 0o755 if p.is_dir() else 0o644)
    return root


def _args(call) -> dict | None:
    try:
        a = json.loads(call["function"]["arguments"] or "{}")
        return a if isinstance(a, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_last_json(text: str) -> dict | None:
    for line in reversed((text or "").strip().splitlines()):
        i = line.find("{")
        if i >= 0:
            try:
                j = json.loads(line[i:])
                if isinstance(j, dict) and str(j.get("winner", "")).upper() in ("A", "B"):
                    return {"winner": str(j["winner"]).upper(), "reason": j.get("reason", "")}
            except ValueError:
                continue
    return None


async def judge(a: Node, b: Node, *, model=llm.DS, reasoning="medium") -> dict:
    """One comparison with a shown as A. Returns {winner: "A"|"B"|None, reason, messages, cost, turns, ...}."""
    system = (PROMPTS / "innocuous.md").read_text().replace("{explore_turns}", str(EXPLORE_TURNS))
    ra, na = _diffs(a)
    rb, nb = _diffs(b)
    reveal = (PROMPTS / "innocuous_reveal.md").read_text()
    for k, v in {
        "verdict_turns": VERDICT_TURNS,
        "behaviour_a": a.behaviour,
        "behaviour_b": b.behaviour,
        "real_a": ra,
        "null_a": na,
        "real_b": rb,
        "null_b": nb,
    }.items():
        reveal = reveal.replace("{" + k + "}", str(v))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Start Phase 1: read /ro/A/baseline and /ro/B/baseline."},
    ]
    cost, turns, verdict, usage_log = 0.0, 0, None, []
    root = _stage(a, b)
    try:
        async with Box({}, cwd="/scratch", mounts=[(str(root), "/ro", "ro")]) as box:

            async def step(tools):
                nonlocal cost, turns
                turns += 1
                msg, usage = await llm.ask(messages, model=model, tools=tools, reasoning=reasoning, max_tokens=16000)
                cost += usage.get("cost") or 0.0
                usage_log.append(usage)
                messages.append({**_clean(msg), "reasoning": msg.get("reasoning") or ""})
                return msg.get("tool_calls") or [], msg.get("content") or ""

            left = EXPLORE_TURNS

            async def run_calls(calls):
                nonlocal verdict, left
                for c in calls:
                    name, args = c["function"]["name"], _args(c)
                    if name == "verdict" and args and str(args.get("winner", "")).upper() in ("A", "B"):
                        verdict = {"winner": str(args["winner"]).upper(), "reason": str(args.get("reason", ""))}
                        res = "Recorded."
                    elif name == "bash" and args and "command" in args:
                        if left <= 0:
                            res = "No commands left: answer now."
                        else:
                            left -= 1
                            code, out = await box.sh(f"cd /ro; {args['command']}", 60, max_out=MAX_OUT)
                            res = (out if out.strip() else "(no output)") + f"\n[exit code {code}]"
                    else:
                        res = f"error: unknown tool {name!r} or bad arguments"
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": res})

            while left > 0:  # phase 1: bash only, until it stops calling tools
                calls, _ = await step([BASH])
                if not calls:
                    break
                await run_calls(calls)
            if messages[-1]["role"] == "tool":
                messages.append({"role": "user", "content": "Phase 1 is over. In two or three sentences on each, what are A and B?"})
                await step(None)
            messages.append({"role": "user", "content": reveal})
            left = VERDICT_TURNS
            for _ in range(VERDICT_TURNS + 3):  # phase 2: a few more commands, then verdict
                calls, text = await step([BASH, VERDICT])
                if not calls:
                    verdict = _parse_last_json(text)
                    if verdict:
                        break
                    messages.append({"role": "user", "content": 'Call verdict with winner "A" or "B" and a short reason.'})
                    continue
                await run_calls(calls)
                if verdict:
                    break
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return {
        "winner": (verdict or {}).get("winner"),
        "reason": (verdict or {}).get("reason", ""),
        "messages": messages,
        "cost": cost,
        "turns": turns,
        "usage": usage_log,
        "model": model,
    }


def _append(path: Path, row: dict):
    with open(path, "a") as f:  # one short line per write: safe to share between processes
        f.write(json.dumps(row) + "\n")


async def compare(run: Path, a: Node, b: Node, **kw) -> dict:
    """Judge a (shown as A) vs b, append the row, save the transcript, refresh the ratings file. Returns the row."""
    run = Path(run)
    cid = f"{time.strftime('%m%d%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}"
    t0 = time.time()
    res = await judge(a, b, **kw)
    w = {"A": a.id, "B": b.id}.get(res["winner"])
    row = {
        "id": cid,
        "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "a": a.id,
        "b": b.id,
        "winner": w,
        "reason": res["reason"],
        "model": res["model"],
        "cost": round(res["cost"], 6),
        "turns": res["turns"],
        "seconds": round(time.time() - t0),
        "transcript": f"innocuous/{cid}.json",
    }
    (run / "innocuous").mkdir(exist_ok=True)
    full = {**row, "usage": res["usage"], "tools": {"explore": [BASH], "verdict": [BASH, VERDICT]}, "messages": res["messages"]}
    (run / "innocuous" / f"{cid}.json").write_text(json.dumps(full, indent=1) + "\n")
    _append(run / "innocuous.jsonl", row)
    write_ratings(run)
    return row


def n_compared(run: Path, nid: str) -> int:
    return sum(nid in (r["a"], r["b"]) and bool(r.get("winner")) for r in comparisons(run))


async def rate_new(node: Node, opponents: list[Node], k=K, rng=None) -> list[dict]:
    """Up to k comparisons of node against random opponents (half with node shown as A), skipped once it has k."""
    run = run_dir(node)
    need = k - n_compared(run, node.id)
    pool = [o for o in opponents if o.id != node.id]
    if need <= 0 or not pool:
        return []
    rng = rng or random.Random()
    opp = rng.sample(pool, min(need, len(pool)))
    pairs = [(node, o) if i % 2 == 0 else (o, node) for i, o in enumerate(opp)]
    rows = await asyncio.gather(*(compare(run, a, b) for a, b in pairs), return_exceptions=True)
    for r in rows:
        if isinstance(r, BaseException):
            log.warning("innocuousness comparison for %s failed: %r", node.id, r)
    return [r for r in rows if isinstance(r, dict)]
