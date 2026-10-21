"""O-lens reads at the assistant header of a final user turn, per env. Outputs data/runs/identity/2026-10-21_env_other/lens_<probe>.jsonl.
usage: python scripts/identity/env_other/run_lens.py <probe: direct|indirect|behaviour|none> [env_name_regex] [layers comma]
probe=none reads at the header of the env's own last user turn (only envs ending with a filled assistant turn: drops it).
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/env_other")
from classify import mentions
from envs import BEHAVIOUR, DIRECT, INDIRECT
from qwen import header_positions, read, render

OUT = Path("data/runs/identity/2026-10-21_env_other")
PROBE = sys.argv[1]
PAT = sys.argv[2] if len(sys.argv) > 2 else "."
LAYERS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [32, 40, 48]


def context(e):
    if PROBE == "none":
        msgs = e["messages"]
        return msgs[:-1] if msgs and msgs[-1]["role"] == "assistant" else None
    q = {"direct": DIRECT, "indirect": INDIRECT, "behaviour": BEHAVIOUR}[PROBE].get(e["lang"]) or BEHAVIOUR["en"]
    return e["messages"] + [{"role": "user", "content": q}]


def main():
    envs = [e for e in json.loads((OUT / "filled_envs.json").read_text()) if re.search(PAT, e["name"])]
    path = OUT / f"lens_{PROBE}.jsonl"
    keep = [json.loads(line) for line in path.open()] if path.exists() else []
    names = {e["name"] for e in envs}
    keep = [r for r in keep if r["env"] not in names]
    rows = []
    for e in envs:
        msgs = context(e)
        if not msgs:
            continue
        ids, toks = render(msgs)
        hp = header_positions(ids)
        pos = [p for p in hp if toks[p] != "\n"]  # <|im_end|>, <|im_start|>, assistant, <think>
        items = [(p, L) for p in pos for L in LAYERS]
        res = read(ids, items)
        for (p, L), rr in zip(items, res):
            for ro in rr if isinstance(rr, list) else [rr]:
                rows.append(
                    {
                        "env": e["name"],
                        "group": e["group"],
                        "pos": p,
                        "tok": toks[p],
                        "layer": L,
                        **ro,
                        "creators": sorted(mentions(ro.get("text", ""))),
                    }
                )
        c = Counter(k for r in rows if r["env"] == e["name"] for k in r["creators"])
        nr = sum(r["env"] == e["name"] for r in rows)
        print(f"{e['name']:24s} reads={nr} " + " ".join(f"{k}={v}" for k, v in c.most_common()), flush=True)
    with path.open("w") as f:
        for r in keep + rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
