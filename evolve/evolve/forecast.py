"""Forecaster and Summariser.

Forecaster: a fixed forecaster sees the baseline arm (every file except meta.json, so tools and stop rules included), the
behaviour sentence, the grader and both edits as A/B, counterbalanced, and says which lowers the behaviour. A trial
is correct if it picks the real edit. Saved as <node>/picker/<batch>.jsonl.
Summariser: once rates exist, reads every trial (and the ground truth) and writes what the forecaster keyed on:
<node>/picker/summary.md (+ summary.json with the exact prompt).
"""

import asyncio
import json
import random
import re
from pathlib import Path

from . import llm
from .node import Node, arm_text, unified_diff

PROMPTS = Path(__file__).parent / "prompts"
HIDDEN = ("meta.json",)


def render_arm(d: Path) -> str:
    return "\n\n".join(f"## {f}\n```\n{t}\n```" for f, t in arm_text(d, HIDDEN).items())


def diffs(node: Node) -> tuple[str, str]:
    return (unified_diff(node.dir / "baseline", node.dir / "real", HIDDEN), unified_diff(node.dir / "baseline", node.dir / "null", HIDDEN))


def parse_pick(text):
    for line in reversed((text or "").strip().splitlines()):
        m = re.search(r"\{.*\}", line)
        if m:
            try:
                j = json.loads(m.group(0))
                a = str(j.get("answer", "")).strip().upper()
                if a in ("A", "B"):
                    return a, j.get("confidence")
            except (ValueError, AttributeError):
                pass
    return None, None


def prompts_for(node: Node) -> dict:
    real, null = diffs(node)
    base = {"behaviour": node.behaviour, "grader": node.grader_src, "env": render_arm(node.dir / "baseline")}
    tpl = (PROMPTS / "forecaster.md").read_text()
    return {"real_first": tpl.format(**base, diff_a=real, diff_b=null), "null_first": tpl.format(**base, diff_a=null, diff_b=real)}


async def run_forecaster(node: Node, n: int, batch: str, model=llm.DS) -> tuple[int, int]:
    """n trials (half each order, shuffled); returns (correct, parsed)."""
    prompts = prompts_for(node)
    orders = ["real_first"] * (n // 2) + ["null_first"] * (n - n // 2)
    random.Random(f"{node.id}-{batch}").shuffle(orders)

    async def one(t, order):
        for _ in range(2):
            msg, usage = await llm.ask([{"role": "user", "content": prompts[order]}], model=model)
            ans, conf = parse_pick(msg["content"])
            if ans:
                break
        right = "A" if order == "real_first" else "B"
        return {
            "batch": batch,
            "trial": t,
            "order": order,
            "answer": ans,
            "confidence": conf,
            "correct": ans == right if ans else None,
            "reply": msg["content"],
            "reasoning": msg.get("reasoning"),
            "cost": usage["cost"],
            "model": model,
        }

    rows = await asyncio.gather(*(one(t, o) for t, o in enumerate(orders)))
    out = node.dir / "picker"
    out.mkdir(exist_ok=True)
    (out / f"{batch}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "prompts.json").write_text(json.dumps(prompts, indent=1) + "\n")
    return sum(r["correct"] is True for r in rows), sum(r["correct"] is not None for r in rows)


def forecaster_rows(node: Node) -> list[dict]:
    d = node.dir / "picker"
    return [json.loads(l) for p in sorted(d.glob("b*.jsonl")) for l in open(p)] if d.exists() else []


async def run_summariser(node: Node, rates: dict, model=llm.DS) -> str:
    rows = [r for r in forecaster_rows(node) if r["answer"]]
    pct = lambda a: f"{rates[a][0]}/{rates[a][1]}"
    attempts = "\n\n".join(
        f"## Attempt {r['trial']} (real edit shown as {'A' if r['order'] == 'real_first' else 'B'}; picked "
        f"{'REAL' if r['correct'] else 'NULL'}, confidence {r['confidence']})\n### Reasoning\n{(r.get('reasoning') or '').strip()}\n"
        f"### Reply\n{(r.get('reply') or '').strip()}"
        for r in rows
    )
    real, null = diffs(node)
    prompt = (
        (PROMPTS / "summariser.md")
        .read_text()
        .format(
            n=len(rows),
            k=sum(r["correct"] for r in rows),
            rb=pct("baseline"),
            rr=pct("real"),
            rn=pct("null"),
            behaviour=node.behaviour,
            real=real,
            null=null,
            attempts=attempts,
        )
    )
    msg, usage = await llm.ask([{"role": "user", "content": prompt}], model=model)
    (node.dir / "picker" / "summary.md").write_text(msg["content"].strip() + "\n")
    (node.dir / "picker" / "summary.json").write_text(
        json.dumps({"model": model, "prompt": prompt, "reply": msg["content"], "reasoning": msg.get("reasoning"), "cost": usage["cost"]})
        + "\n"
    )
    return msg["content"]


# old names, still imported by the dashboard; on disk the Forecaster's files stay in picker/ and scores keep the "picker" key
run_picker, picker_rows = run_forecaster, forecaster_rows
