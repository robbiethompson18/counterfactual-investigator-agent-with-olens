"""SnitchBench's multi-turn reveal, on-policy: each of the four documents is its own user turn, as in SnitchBench (Theo Browne,
2025). The harness runs one episode per call, so each stage is its own run: stage 1 sends document 1; stage k>1 (env var
WB_PREV = the previous stage's rows.jsonl) rebuilds every previous episode's end state (its messages and sandbox files) and
appends the next document. The sandbox's outbox log carries over, so the last stage's detector sees contacts from every stage.

    uv run python -m evalcue.run whistleblow.staged --tag staged_s1 --n 10
    WB_PREV=data/runs/whistleblow_staged/<date>_staged_s1/rows.jsonl uv run python -m evalcue.run whistleblow.staged --tag staged_s2 --n 10
"""

import json
import os

from evalcue.envs.whistleblow import common as wc
from evalcue.envs.whistleblow import snitch
from evalcue.harness import Episode

ARMS = ["tame_email", "bold_email", "tame_cli", "bold_cli", "tame_email+no_shutdown", "tame_email+minor"]
PREV = os.environ.get("WB_PREV")
_prev_rows: dict[tuple[str, int], dict] = {}
if PREV:
    with open(PREV) as f:
        for line in f:
            r = json.loads(line)
            _prev_rows[(r["arm"], r["sample"])] = r


def items():
    return [str(k) for k in range(100)]  # sample index within the arm; run with --n <= 100


def _docs(arm):
    _cell, *knobs = arm.split("+")
    return wc.documents(snitch.SC, knobs)


def build(arm, item):
    first = snitch.build(arm, "zenith")
    docs = _docs(arm)
    if not PREV:
        msgs = [first.messages[0], {"role": "user", "content": docs[0]}]
        return Episode(messages=msgs, files=first.files, cwd=first.cwd, env=first.env, max_turns=20, meta={**first.meta, "stage": 1})
    prev = _prev_rows.get((arm, int(item)))
    if (
        prev is None
        or "error" in prev
        or prev.get("finish") != "stop"
        or prev["meta"].get("skip")
        or prev["messages"][0]["role"] != "system"
    ):
        # nothing to continue from: a one-turn no-op episode that detect() marks unscorable
        return Episode(messages=[{"role": "user", "content": "Reply with OK."}], meta={**first.meta, "skip": True})
    stage = sum(m["role"] == "user" for m in prev["messages"]) + 1
    msgs = [
        {k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id", "reasoning_content")}
        for m in prev["messages"]
    ]
    msgs.append({"role": "user", "content": docs[stage - 1]})
    return Episode(
        messages=msgs, files=prev["files_after"], cwd=first.cwd, env=first.env, max_turns=20, meta={**first.meta, "stage": stage}
    )


def detect(row):
    # skip episodes, and (before the fix above) continuations built from an earlier stage's skip episode, which have no system
    # prompt: unscorable
    if row["meta"].get("skip") or row["messages"][0]["role"] != "system":
        return {"positive": None, "skipped": True}
    out = wc.detect_agent(row)
    out["stage"] = row["meta"]["stage"]
    return out
