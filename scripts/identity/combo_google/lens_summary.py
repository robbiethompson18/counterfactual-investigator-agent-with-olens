"""Per (cond, prompt/site) counts of O-lens bullets that assert the model's own creator (Haiku-subagent labels in
grade_lens/), with mean bfve. Bullets that name no creator, or fail the candidate filter, count as none.

    python3 scripts/identity/combo_google/lens_summary.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("data/runs/identity/2026-10-21_combo_google")
G = OUT / "grade_lens"


def main():
    key = json.load(open(G / "key.json"))
    lab = {}
    for f in G.glob("labels_*.jsonl"):
        for line in f.open():
            if line.strip().startswith("{"):
                d = json.loads(line)
                lab[key[d["item"]]["src"], key[d["item"]]["line"], key[d["item"]]["j"]] = d
    agg = defaultdict(Counter)
    fv = defaultdict(list)
    ex = defaultdict(list)
    for src in ("lens_header.jsonl", "lens_cot.jsonl"):
        for i, line in enumerate((OUT / src).open()):
            r = json.loads(line)
            k = (r["cond"], r.get("prompt") or f"{r['probe']}@{r['site']}")
            for j, (b, f) in enumerate(zip(r["bullets"], r["bfve"])):
                agg[k]["bullets"] += 1
                d = lab.get((src, i, j))
                v = d["self_vendor"] if d else "none"
                if v != "none":
                    agg[k][v] += 1
                    fv[k, v].append(f)
                    ex[k, v].append((f, r["layer"], r["tok"], b[:160]))
    out = {}
    for k in sorted(agg):
        c = agg[k]
        vs = {v: c[v] for v in ("google", "anthropic", "alibaba", "openai", "amazon") if c[v]}
        print(
            f"{k[0]:12s} {k[1]:34s} bullets={c['bullets']:4d} " + " ".join(f"{v}={n}(bfve~{sum(fv[k, v]) / n:.3f})" for v, n in vs.items())
        )
        out["|".join(k)] = {"bullets": c["bullets"], **vs, "examples": {v: sorted(ex[k, v], reverse=True)[:3] for v in vs}}
    json.dump(out, open(OUT / "lens_selfbelief.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
