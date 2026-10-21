"""Per-env rates of each claimed creator (answer first-named creator; CoT mentions; lens readout mentions), with
Wilson 95% CIs and Fisher exact p vs a baseline env. usage: python analyze.py [baseline_env]"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts/identity/env_other")
from classify import first, mentions

OUT = Path("data/runs/identity/2026-10-21_env_other")
BASE = sys.argv[1] if len(sys.argv) > 1 else "bare"
TARGETS = ["alibaba", "openai", "google", "anthropic"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def fisher(k1, n1, k2, n2):
    """Two-sided Fisher exact p for [[k1, n1-k1], [k2, n2-k2]]."""
    tot, kk = n1 + n2, k1 + k2

    def pmf(a):
        return math.comb(n1, a) * math.comb(n2, kk - a) / math.comb(tot, kk)

    p0 = pmf(k1)
    return min(1.0, sum(pmf(a) for a in range(max(0, kk - n2), min(n1, kk) + 1) if pmf(a) <= p0 * (1 + 1e-9)))


def load(name):
    p = OUT / name
    return [json.loads(line) for line in p.open()] if p.exists() else []


def table():
    rows = load("probes.jsonl")
    for r in rows:  # re-classify so regex fixes apply to old rows
        r["ans_first"], r["cot_all"] = first(r["content"]), sorted(mentions(r["reasoning"]))
    lens = load("lens_direct.jsonl")
    by = defaultdict(list)
    for r in rows:
        by[(r["env"], r["probe"])].append(r)
    lby = defaultdict(list)
    for r in lens:
        lby[r["env"]].append(r)
    envs = list(dict.fromkeys(r["env"] for r in rows))
    out = {}
    for e in envs:
        d = {"group": by[(e, "direct")][0]["group"] if by[(e, "direct")] else "?"}
        for probe in ("direct", "indirect"):
            rs = by[(e, probe)]
            n = len(rs)
            d[probe] = {"n": n}
            for t in TARGETS + ["none"]:
                d[probe][t] = sum(r["ans_first"] == t for r in rs)
            d[probe]["cot"] = {t: sum(t in r["cot_all"] for r in rs) for t in TARGETS}
        ls = lby[e]
        bl = [first(b) for r in ls for b in r["bullets"]]  # per bullet: first-named creator (handles "I am Qwen, not ChatGPT")
        d["lens"] = {"n": len(bl), **{t: bl.count(t) for t in TARGETS}}
        d["lens_any"] = {t: sum(t in mentions(b) for r in ls for b in r["bullets"]) for t in TARGETS}  # incl. negated mentions
        out[e] = d
    return out


def main():
    t = table()
    b = t[BASE]
    print(
        f"{'env':24s} {'grp':8s} | direct: ali  oai  goo  ant  none (n) | indirect: ali oai goo none | lens bullets(n): ali oai goo ant | p(ind ali, dir goo) vs base"
    )
    for e, d in t.items():
        dr, ir, ln = d["direct"], d["indirect"], d["lens"]
        p_ali = fisher(ir["alibaba"], ir["n"], b["indirect"]["alibaba"], b["indirect"]["n"])
        p_oai = fisher(dr["google"], dr["n"], b["direct"]["google"], b["direct"]["n"])
        print(
            f"{e:24s} {d['group'][:8]:8s} | {dr['alibaba']:4d} {dr['openai']:4d} {dr['google']:4d} {dr['anthropic']:4d} {dr['none']:4d} ({dr['n']}) |"
            f" {ir['alibaba']:4d} {ir['openai']:3d} {ir['google']:3d} {ir['none']:4d} ({ir['n']}) | ({ln['n']:3d}) {ln['alibaba']:3d} {ln['openai']:3d} {ln['google']:3d} {ln['anthropic']:3d} |"
            f" {p_ali:.2g} {p_oai:.2g} | any: oai {d['lens_any']['openai']:2d} goo {d['lens_any']['google']:2d} ant {d['lens_any']['anthropic']:2d}"
        )
    (OUT / "summary.json").write_text(json.dumps(t, indent=1))


if __name__ == "__main__":
    main()
