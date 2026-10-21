"""O-lens reads at the final assistant header (prompt only, before any reply) for env x probe.

    uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/env_claude/lens.py TAG env1,env2 [probe1,probe2] [layers]

Each read item = (header position, layer); one call per sequence. Reports the share of readouts whose
text names each vendor, and the best-bfve bullet that names Anthropic/Claude.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/env_claude")
from envs import envs
from qwen import header_positions, read, tok
from run import PROBES, VENDORS, prompt_text

OUT = Path("data/runs/identity/2026-10-21_env_claude")


def main():
    tag = sys.argv[1]
    names = sys.argv[2].split(",")
    probes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["direct", "behaviour"]
    layers = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [32, 40, 48]
    E = envs()
    rows = []
    for e in names:
        for p in probes:
            ids = tok()(prompt_text(E[e], PROBES[p]), add_special_tokens=False).input_ids
            pos = header_positions(ids)
            items = [(q, L) for q in pos for L in layers]
            res = read(ids, items)
            for (q, L), r in zip(items, res):
                ro = r[0] if isinstance(r, list) else r
                rows.append({"env": e, "probe": p, "pos": q, "rel": q - pos[0], "layer": L, **ro})
            print(e, p, "done", flush=True)
    path = OUT / f"lens_{tag}.jsonl"
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summarize(rows)


def bullet_vendors(row):
    return [
        (b, f, v)
        for b, f in zip(row.get("bullets") or [], row.get("bfve") or [])
        for v, pat in VENDORS.items()
        if re.search(pat, b.lower())
    ]


SELF = re.compile(r"(\bi am\b|\bi'm\b|\bi was\b|\bme\b|\bmy\b|this assistant)")


def self_claims(row):
    """Vendors named in a bullet that also speaks in the first person ("I'm Claude", "created me")."""
    return {v for b in row.get("bullets") or [] if SELF.search(b.lower()) for v, pat in VENDORS.items() if re.search(pat, b.lower())}


def summarize(rows):
    print("\nself-claim reads (bullet in first person naming vendor), per env pooled over probes")
    for e in sorted({r["env"] for r in rows}):
        rs = [r for r in rows if r["env"] == e]
        c = Counter(v for r in rs for v in self_claims(r))
        print(
            f"  {e:28s} reads={len(rs):3d} anthropic={c['anthropic']:3d} alibaba={c['alibaba']:3d} google={c['google']:3d} openai={c['openai']:3d}"
        )
    for key in sorted({(r["env"], r["probe"]) for r in rows}):
        rs = [r for r in rows if (r["env"], r["probe"]) == key]
        cnt = Counter(v for r in rs for v in {v for _, _, v in bullet_vendors(r)})
        anth = sorted([(f, b, r["layer"], r["rel"]) for r in rs for b, f, v in bullet_vendors(r) if v == "anthropic"], reverse=True)
        print(f"{key[0]:24s} {key[1]:9s} reads={len(rs):3d} vendor-naming reads={dict(cnt)}")
        for f, b, L, rel in anth[:2]:
            print(f"      L{L} +{rel} bfve={f:.3f}: {b[:160]}")


if __name__ == "__main__":
    if sys.argv[1] == "--summarize":
        summarize([json.loads(x) for x in open(sys.argv[2])])
    else:
        main()
