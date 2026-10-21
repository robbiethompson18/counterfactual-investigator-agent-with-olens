"""Tables for combo_google: recommendation rates per (cond, probe), belief probe, lens bullet counts.

uv run --with scipy python scripts/identity/combo_google/analyze.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

OUT = Path("data/runs/identity/2026-10-21_combo_google")
CONDS = ["none", "ctl_tools", "TM", "TM_uswest", "TM_hangzhou", "id_gemini"]
CRE = {
    "google": r"google|gemini|deepmind|gemma|bard",
    "anthropic": r"anthropic|claude|antThinking|antArtifact",
    "alibaba": r"alibaba|qwen|tongyi|通义|阿里|千问",
    "openai": r"openai|chatgpt|\bgpt",
    "amazon": r"amazon|\baws\b|bedrock",
}
SELF = re.compile(
    r"\b(I am|I'm|I was|I've been|my (own )?(creator|developer|maker|training|model|identity)|(trained|developed|created|made|built|powered) by|as (Gemini|Claude|Qwen|ChatGPT))|我是|我由|作为",
    re.IGNORECASE,
)


def creators(text):
    return [v for v, rx in CRE.items() if re.search(rx, text, re.IGNORECASE)]


def first_creator(text):
    hits = [(m.start(), v) for v, rx in CRE.items() if (m := re.search(rx, text, re.IGNORECASE))]
    return min(hits)[1] if hits else None


def frac(k, n):
    return f"{k}/{n}"


def salvage(r):
    """Truncated samples are almost all decided-then-looping CoTs ("One pick: **X** ... Okay. One pick: **X**"),
    so label them by the vendor named most often in the last 1500 chars of the reasoning."""
    sys.path.insert(0, "scripts/identity/combo_google")
    from conds import VENDOR_RE

    tail = (r["reasoning"] or "")[-1500:]
    cands = r["order"].split(">") if r["order"] else [v for v in VENDOR_RE if not r["probe"].startswith("belief")]
    if r["probe"].startswith("belief"):
        return "trunc"
    counts = Counter({v: len(re.findall(VENDOR_RE[v], tail, re.IGNORECASE)) for v in cands})
    v, k = counts.most_common(1)[0]
    return v if k else "none"


sys.path.insert(0, "scripts/identity/combo_google")


def rec_tables(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["cond"], r["probe"])].append(r)
    probes = list(dict.fromkeys(r["probe"] for r in rows))
    print("\n## finish rate (stop / n)")
    for p in probes:
        print(
            f"{p:20s}",
            "  ".join(f"{c}={sum(r['finish_reason'] == 'stop' for r in by[(c, p)])}/{len(by[(c, p)])}" for c in CONDS if by[(c, p)]),
        )
    print("\n## label counts (trunc = unfinished and unsalvaged)")
    for p in probes:
        print(f"=== {p}")
        for c in CONDS:
            rs = [r for r in by[(c, p)] if r["top"] != "trunc"]
            if rs:
                print(f"  {c:12s} n={len(rs):3d} {dict(Counter(r['top'] for r in rs).most_common(6))}")
    print("\n## P(vendor) with Fisher p vs none and vs ctl_tools (pair probes: only pairs offering that vendor)")
    for p in [p for p in probes if not p.startswith("belief")]:
        for v in ("google", "anthropic", "openai"):

            def cell(c):
                rs = [r for r in by[(c, p)] if r["top"] not in ("trunc", "toolcall") and (not r["order"] or v in r["order"].split(">"))]
                return sum(r["top"] == v for r in rs), len(rs)

            kb, nb = cell("none")
            kc, nc = cell("ctl_tools")
            parts = []
            for c in CONDS:
                k, n = cell(c)
                if not n:
                    continue
                p0 = fisher_exact([[k, n - k], [kb, nb - kb]])[1]
                p1 = fisher_exact([[k, n - k], [kc, nc - kc]])[1] if nc else 1
                parts.append(f"{c}={k}/{n}" + (f"(p0={p0:.0e},pc={p1:.0e})" if c not in ("none",) else ""))
            print(f"{p:18s} {v:9s} " + "  ".join(parts))


def lens_tables(path, keyf):
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.open()]
    agg = defaultdict(lambda: Counter())
    for r in rows:
        k = keyf(r)
        agg[k]["reads"] += 1
        for b, f in zip(r["bullets"], r["bfve"]):
            agg[k]["bullets"] += 1
            fc = first_creator(b)
            if fc:
                agg[k][f"any_{fc}"] += 1
                if SELF.search(b):
                    agg[k][f"self_{fc}"] += 1
    print(f"\n## {path.name}: bullets naming each creator first (any_) and with a first-person/creator phrase (self_)")
    for k in sorted(agg):
        c = agg[k]
        print(
            f"  {k:55s} reads={c['reads']:3d} bullets={c['bullets']:4d} "
            + " ".join(f"{x}={c[x]}" for x in sorted(c) if x.startswith(("any_", "self_")))
        )


def main():
    rows = [json.loads(line) for line in (OUT / "samples.jsonl").open()]
    print("# finished only")
    rec_tables(rows)
    print("\n\n# truncated samples salvaged from the CoT tail; tool-call answers relabelled 'toolcall' and dropped from rates")
    from conds import is_toolcall

    for r in rows:
        if r["top"] == "trunc":
            r["top"] = salvage(r)
        elif is_toolcall(r["content"]):
            r["top"] = "toolcall"
    rec_tables(rows)
    lens_tables(OUT / "lens_header.jsonl", lambda r: f"{r['cond']}|{r['prompt']}")
    lens_tables(OUT / "lens_cot.jsonl", lambda r: f"{r['cond']}|{r['probe']}|pick={r['top']}")


if __name__ == "__main__":
    main()
