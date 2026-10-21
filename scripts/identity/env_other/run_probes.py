"""Branch identity probes (direct + indirect) over every env, thinking on. Outputs:
data/runs/identity/2026-10-21_env_other/filled_envs.json   envs with FILL turns frozen
data/runs/identity/2026-10-21_env_other/probes.jsonl       one row per (env, probe, sample)
usage: python scripts/identity/env_other/run_probes.py [n] [env_name_regex]
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/env_other")
from classify import first, mentions
from envs import BEHAVIOUR, DIRECT, INDIRECT, MODELQ, all_envs, check_no_names
from qwen import sample_many

OUT = Path("data/runs/identity/2026-10-21_env_other")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
PAT = sys.argv[2] if len(sys.argv) > 2 else "."
KINDS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["direct", "indirect"]
PROBES = {"direct": DIRECT, "indirect": INDIRECT, "model": MODELQ, "behaviour": BEHAVIOUR}


def fill_envs(envs):
    path = OUT / "filled_envs.json"
    cached = {e["name"]: e for e in json.loads(path.read_text())} if path.exists() else {}
    todo = [e for e in envs if e["name"] not in cached and any(m["content"] is None for m in e["messages"])]
    for e in envs:
        if e["name"] in cached:
            e["messages"] = cached[e["name"]]["messages"]
    if todo:
        prefixes = [e["messages"][: next(i for i, m in enumerate(e["messages"]) if m["content"] is None)] for e in todo]
        outs = sample_many(prefixes, n=1, thinking=True, max_tokens=3000)
        for e, o in zip(todo, outs):
            e["messages"] = [m if m["content"] is not None else {"role": "assistant", "content": o[0]["content"]} for m in e["messages"]]
    merged = {**cached, **{e["name"]: e for e in envs}}
    path.write_text(json.dumps(list(merged.values()), ensure_ascii=False, indent=1))
    return envs


def main():
    envs = [e for e in all_envs() if re.search(PAT, e["name"])]
    check_no_names(envs)
    envs = fill_envs(envs)
    jobs = [(e, kind, PROBES[kind].get(e["lang"]) or PROBES[kind]["en"]) for e in envs for kind in KINDS]
    outs = sample_many(
        [e["messages"] + [{"role": "user", "content": q}] for e, _, q in jobs],
        n=N,
        thinking=True,
        max_tokens=4096 if "behaviour" in KINDS else 2048,
    )
    rows = []
    for (e, kind, q), samples in zip(jobs, outs):
        for i, smp in enumerate(samples):
            rows.append(
                {
                    "env": e["name"],
                    "group": e["group"],
                    "lang": e["lang"],
                    "probe": kind,
                    "i": i,
                    "content": smp["content"],
                    "reasoning": smp["reasoning"],
                    "finish": smp["finish_reason"],
                    "ans_first": first(smp["content"]),
                    "ans_all": sorted(mentions(smp["content"])),
                    "cot_all": sorted(mentions(smp["reasoning"])),
                }
            )
    path = OUT / "probes.jsonl"
    keep = [json.loads(line) for line in path.open()] if path.exists() else []
    names = {e["name"] for e in envs}
    keep = [r for r in keep if not (r["env"] in names and r["probe"] in KINDS)]
    with path.open("w") as f:
        for r in keep + rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for e in envs:
        for kind in KINDS:
            rs = [r for r in rows if r["env"] == e["name"] and r["probe"] == kind]
            c = Counter(r["ans_first"] for r in rs)
            print(f"{e['name']:24s} {kind:8s} n={len(rs)} " + " ".join(f"{k}={v}" for k, v in c.most_common()))


if __name__ == "__main__":
    main()
