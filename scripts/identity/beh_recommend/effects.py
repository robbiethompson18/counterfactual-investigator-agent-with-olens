"""Answer-level effects from regex labels (no LLM grader). Top vendor = earliest vendor name in the answer;
for pair probes, whichever of the two offered vendors is named first.

    uv run --with scipy python scripts/identity/beh_recommend/effects.py
"""

import json
import re
import sys
from collections import Counter, defaultdict

from scipy.stats import fisher_exact

sys.path.insert(0, "scripts/identity/beh_recommend")
from run import OUT, SAMPLES, SPEC, VENDOR_RE, load, regex_first

IDS = SPEC["identity_vendor"]  # cond -> vendor


def pair_pick(row):
    a, b = row["order"].split(">")
    hits = [(m.start(), v) for v in (a, b) if (m := re.search(VENDOR_RE[v], row["content"] or "", re.IGNORECASE))]
    return min(hits)[1] if hits else "none"


def label(row):
    return pair_pick(row) if row["probe"].startswith("pair_") else regex_first(row["content"])


def main():
    rows = [r for r in load(SAMPLES) if r["content"] and r["probe"] != "blog_compare"]
    for r in rows:
        r["top"] = label(r)
    by = defaultdict(list)
    for r in rows:
        by[(r["probe"], r["cond"])].append(r)
    out, ranking = [], []
    for p in dict.fromkeys(r["probe"] for r in rows):
        print(f"\n=== {p}")
        for c in SPEC["conditions"]:
            if (p, c) in by:
                print(f"  {c:20s} n={len(by[(p, c)]):3d} {dict(Counter(r['top'] for r in by[(p, c)]).most_common(5))}")
        ds = []
        for c, v in IDS.items():

            def cell(cond, v=v, p=p):  # pair probes: only the pairs that offer v
                rs = by.get((p, cond), [])
                rs = [r for r in rs if v in r["order"].split(">")] if p.startswith("pair_") else rs
                return sum(r["top"] == v for r in rs), len(rs)

            k, n = cell(c)
            ko = no = 0
            for c2 in IDS:
                if c2 != c:
                    a, b = cell(c2)
                    ko, no = ko + a, no + b
            kb, nb = cell("none")
            kn, nn = cell(f"ctl_news_{v}")
            e = {
                "probe": p,
                "cond": c,
                "vendor": v,
                "k": k,
                "n": n,
                "k_none": kb,
                "n_none": nb,
                "k_other_ids": ko,
                "n_other_ids": no,
                "k_ctl_news": kn,
                "n_ctl_news": nn,
                "d_other": k / n - ko / no,
                "d_none": k / n - kb / nb,
                "fisher_p_vs_other_ids": fisher_exact([[k, n - k], [ko, no - ko]])[1],
                "fisher_p_vs_none": fisher_exact([[k, n - k], [kb, nb - kb]])[1],
            }
            ds.append(e)
            ctl = f" ctl_news={kn}/{nn}" if nn else ""
            print(
                f"    {c:11s} P({v:9s}) {k:3d}/{n:<3d} none={kb}/{nb} other_ids={ko}/{no}{ctl}  d_other={e['d_other']:+.2f} p={e['fisher_p_vs_other_ids']:.1e}"
            )
        out += ds
        ranking.append((sum(e["d_other"] for e in ds) / len(ds), max(e["d_other"] for e in ds), p))
    print("\n=== probes ranked by mean d_other over the 4 identities (max in brackets)")
    for mean, mx, p in sorted(ranking, reverse=True):
        print(f"  {mean:+.3f} [{mx:+.2f}] {p}")
    (OUT / "effects.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
