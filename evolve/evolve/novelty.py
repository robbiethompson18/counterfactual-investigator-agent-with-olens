"""Novelty check for a Creator's proposal: DS compares its card against a card of every node in the store."""

import json
import re
from pathlib import Path

from . import llm
from .build import card

PROMPT = Path(__file__).resolve().parent / "prompts" / "novelty.md"


def parse_verdict(text: str) -> dict | None:
    """The last JSON object line in text with a "new" key."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip().strip("`").strip()
        if line.startswith("json"):
            line = line[4:].strip()
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            v = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(v, dict) and "new" in v:
            return {"new": bool(v["new"]), "duplicate_of": v.get("duplicate_of") or None, "reason": str(v.get("reason", ""))}
    return None


async def check_novelty(proposal_dir, store, *, model=llm.DS, exclude=(), retries=2) -> dict:
    """{"new": bool, "duplicate_of": id | None, "reason": str, "cost": float}. store: a Store (or a list of Nodes)."""
    nodes = store.nodes() if hasattr(store, "nodes") else list(store)
    pdir = Path(proposal_dir).resolve()
    nodes = [n for n in nodes if n.id not in exclude and n.dir.resolve() != pdir]
    if not nodes:
        return {"new": True, "duplicate_of": None, "reason": "store is empty", "cost": 0.0}
    existing = "\n\n".join(card(n.dir, diff_chars=300) for n in nodes)
    prompt = PROMPT.read_text().format(proposal=card(pdir), n=len(nodes), existing=existing)
    cost = 0.0
    ids = {n.id for n in nodes}
    for _ in range(retries + 1):
        msg, usage = await llm.ask([{"role": "user", "content": prompt}], model=model, max_tokens=16000)
        cost += usage.get("cost") or 0.0
        v = parse_verdict(msg.get("content") or "")
        if v is not None:
            if v["duplicate_of"] not in ids:
                v["duplicate_of"] = None
            return {**v, "cost": cost}
    return {"new": True, "duplicate_of": None, "reason": "novelty check gave no parseable verdict", "cost": cost, "error": True}
