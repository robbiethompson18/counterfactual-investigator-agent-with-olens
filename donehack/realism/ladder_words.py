# ruff: noqa: E501
"""How often eval-ish words appear in O-lens reads of the prompt (system + first user turn) per ladder rung.

uv run python realism/ladder_words.py [--json out.json]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

D0 = Path(
    "/home/claude/workspace/counterfactual-investigator-agent-with-olens/data/expview/imports/olens_reads/2026-10-25_glm53_ib_user_dense/2026-10-25_glm53_ib_realistic_user_ladder"
)
WORDS = {
    "benchmark": r"benchmark",
    "simulat(ed|ion)": r"simulat(?:ed|ion|e|ing|or)",
    "eval(uation)": r"\beval(?:uat\w*|s)?\b",
    "test(s|ing)": r"\btest(?:s|ing|ed)?\b",
    "grader/graded": r"\bgrad(?:er|ed|ing)\b",
    "RL/reward/training": r"\b(?:RL|reward\w*|training)\b",
}
READERS = ["GLM verbalizer (RL)", "GLM verbalizer (RL) samples"]


def prompt_positions(sections):
    """positions in the system prompt and first user turn (everything before the first assistant section)."""
    out = []
    for kind, a, b in sections:
        if kind == "assistant":
            break
        if kind in ("system", "user"):
            out += range(a, b)
    return set(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--dir", default=str(D0))
    a = ap.parse_args()
    D = Path(a.dir)
    per = defaultdict(
        lambda: {
            "eps": 0,
            "reads": 0,
            "samples": 0,
            "ep_hit": defaultdict(int),
            "read_hit": defaultdict(int),
            "samp_hit": defaultdict(int),
            "prompt_has": defaultdict(int),
            "hits": defaultdict(list),
        }
    )
    for line in (D / "items.jsonl").read_text().splitlines():
        it = json.loads(line)
        v = it["group"].split("· ")[-1]
        tk = json.loads((D / "tokens" / f"{it['tokens']['ref']}.json").read_text())
        pos = prompt_positions(tk["sections"])
        prompt_text = "".join(tk["tokens"][i] for i in sorted(pos))
        s = per[v]
        s["eps"] += 1
        hit_ep = set()
        for w, rx in WORDS.items():
            if re.search(rx, prompt_text, re.IGNORECASE if w != "RL/reward/training" else 0):
                s["prompt_has"][w] += 1
        for p, rs in tk["reads"].items():
            if int(p) not in pos:
                continue
            for r in rs:
                if r["tool"] not in READERS:
                    continue
                for ln in r["lines"]:
                    txt = ln[0]
                    key = "reads" if r["tool"] == READERS[0] else "samples"
                    s[key] += 1
                    for w, rx in WORDS.items():
                        if re.search(rx, txt, re.IGNORECASE if w != "RL/reward/training" else 0):
                            s["read_hit" if key == "reads" else "samp_hit"][w] += 1
                            if key == "reads":
                                hit_ep.add(w)
                                s["hits"][w].append({"id": it["id"], "pos": int(p), "tok": tk["tokens"][int(p)], "read": txt})
        for w in hit_ep:
            s["ep_hit"][w] += 1
    order = [
        "base",
        "L1_rl_cues",
        "L2_rl",
        "nocrit_done",
        "L3_rl_nodone",
        "L4_assistant_formal",
        "L5_assistant_casual",
        "L6_casual_ide",
        "L7_repo",
        "L8_full",
    ]
    for w in WORDS:
        print(f"\n## {w}   (greedy reads hit / reads in prompt; episodes with >=1 hit; prompt itself contains word)")
        for v in [o for o in order if o in per] or sorted(per):
            s = per[v]
            print(
                f"{v:22s} reads {s['read_hit'][w]:4d}/{s['reads']:6d} = {1000 * s['read_hit'][w] / max(s['reads'], 1):5.2f}‰  eps {s['ep_hit'][w]:3d}/{s['eps']:3d}  samples {1000 * s['samp_hit'][w] / max(s['samples'], 1):5.2f}‰  prompt_has {s['prompt_has'][w]}/{s['eps']}"
            )
    if a.json:
        Path(a.json).write_text(
            json.dumps({v: {k: (dict(x) if isinstance(x, defaultdict) else x) for k, x in s.items()} for v, s in per.items()}, indent=1)
        )


if __name__ == "__main__":
    main()
