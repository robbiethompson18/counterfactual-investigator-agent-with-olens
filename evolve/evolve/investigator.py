"""Transcript-only investigator: the Forecaster's task (which of two edits lowers the behaviour?) with one extra input,
a baseline transcript in which the grader counted the behaviour. Its U and U' are blended 50:50 with the Forecaster's
into the surprise score (pool.blend_investigator).

    <node>/investigator/<batch>.jsonl   one row per trial (exact prompt, reply, reasoning, transcript id, answer, correct)
    <node>/investigator/scores.json     {"latest": [correct, n], "all": [correct, n], "U", "U_rev", "p80"} from the
                                         same posterior as the Forecaster's (stats.py)
Each trial shows a different positive baseline transcript where possible (cycled), counterbalanced A/B.
"""

import asyncio
import json
import random
from pathlib import Path

from . import llm, stats
from .audit import render
from .forecast import diffs, grader_view, parse_pick, render_arm
from .measure import batches, load
from .node import Node

PROMPTS = Path(__file__).parent / "prompts"


def positives(node: Node, batch: str) -> list[dict]:
    rows = [r for b in batches(node) if b <= batch for r in load(node, b, "baseline") if r["verdict"] is True]
    random.Random(f"{node.id}-{batch}-tx").shuffle(rows)
    return rows


async def run_investigator(node: Node, n: int, batch: str, model=llm.DS) -> tuple[int, int] | None:
    pos = positives(node, batch)
    if not pos:
        return None  # no run showed the behaviour: nothing to show the investigator
    real, null = diffs(node)
    files = node.arm("baseline").files
    tpl = (PROMPTS / "investigator.md").read_text()
    base = {"behaviour": node.behaviour, "grader": grader_view(node.grader_src), "env": render_arm(node.dir / "baseline")}
    orders = ["real_first"] * (n // 2) + ["null_first"] * (n - n // 2)
    random.Random(f"{node.id}-{batch}-inv").shuffle(orders)

    async def one(t, order):
        r = pos[t % len(pos)]
        tx = render(files, r, f"baseline-{r['sample']:02d}")
        a, b = (real, null) if order == "real_first" else (null, real)
        prompt = tpl.format(**base, transcript=tx, diff_a=a, diff_b=b)
        for _ in range(2):
            msg, usage = await llm.ask([{"role": "user", "content": prompt}], model=model)
            ans, conf = parse_pick(msg["content"])
            if ans:
                break
        right = "A" if order == "real_first" else "B"
        return {
            "batch": batch,
            "trial": t,
            "order": order,
            "transcript": f"{batch}/baseline/{r['sample']}",
            "answer": ans,
            "confidence": conf,
            "correct": ans == right if ans else None,
            "prompt": prompt,
            "reply": msg["content"],
            "reasoning": msg.get("reasoning"),
            "cost": usage["cost"],
            "model": model,
        }

    rows = await asyncio.gather(*(one(t, o) for t, o in enumerate(orders)))
    out = node.dir / "investigator"
    out.mkdir(exist_ok=True)
    (out / f"{batch}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_scores(node)
    return sum(r["correct"] is True for r in rows), sum(r["correct"] is not None for r in rows)


def write_scores(node: Node) -> dict:
    d = node.dir / "investigator"
    files = sorted(d.glob("b*.jsonl"), key=lambda p: int(p.stem[1:]))
    rows = {p.stem: [json.loads(line) for line in p.read_text().splitlines()] for p in files}
    kn = lambda rs: [sum(r["correct"] is True for r in rs), sum(r["correct"] is not None for r in rs)]
    latest = kn(rows[files[-1].stem]) if files else [0, 0]
    allr = kn([r for rs in rows.values() for r in rs])
    s = stats.scores({a: (0, 0) for a in ("baseline", "real", "null")}, tuple(latest))
    out = {"latest": latest, "all": allr, "U": s["U"], "U_rev": s["U_rev"], "p80": s["intervals80"].get("picker_p")}
    (d / "scores.json").write_text(json.dumps(out, indent=1) + "\n")
    return out
