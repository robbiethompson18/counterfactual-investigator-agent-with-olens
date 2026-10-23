"""Novelty of O-lens reads over the realized continuation (olens.novelty), on CPU from atom dumps: the residual-reads
run's atoms/ (scripts/olens_residual_reads.py save_atoms, joined to its reads.jsonl for text and env labels) or
scripts/olens_atoms.py's (which carry their own text).

    uv run python scripts/olens_novelty.py RUN_DIR [--draws 3] [--prefixes 1,2,4,8,16,32]

Per (transcript, position, layer): cont, lens, joint, gain (olens.novelty), two nulls (the gain from the bullets read
at another position of the same transcript, >= 64 tokens away where possible, or of another transcript; mean of
--draws), novelty = gain − null_other, and per bullet n_b, solo FVE and overlap (share of its word 3-grams found in the
32 tokens before + the 32 after the position). --prefixes keeps only those continuation prefix lengths, to compare
with dumps that store fewer.

Writes RUN_DIR/novelty.jsonl and RUN_DIR/novelty_extremes.md (transcripts by mean novelty, then the --extremes highest
and lowest positions with their bullets) and prints a summary per layer and role.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from olens.novelty import null_gain, scores

PLAIN = ("main", 0.0, 1.0, 1.0)  # residual reads: the ordinary read's (arm, beta, gamma, mu_scale)


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def overlap(bullet, context):
    w, ctx = words(bullet), " " + " ".join(words(context)) + " "
    if len(w) < 3:
        return float(f" {' '.join(w)} " in ctx) if w else 0.0
    grams = [" ".join(w[i : i + 3]) for i in range(len(w) - 2)]
    return sum(f" {g} " in ctx for g in grams) / len(grams)


def load(run):
    """{id: atom dict}, plus per-position text/labels {(id, token_index, layer): row} from reads.jsonl when present."""
    files = {}
    for f in sorted((run / "atoms").glob("*.pt")):
        a = torch.load(f)
        files[a["id"]] = a
    rows = {}
    if (run / "reads.jsonl").exists():
        with open(run / "reads.jsonl") as f:
            for r in map(json.loads, f):
                if (r.get("arm"), r.get("beta"), r.get("gamma"), r.get("mu_scale")) == PLAIN:
                    rows[(r["id"], r["token_index"], r["layer"])] = r
    return files, rows


def info(a, rows, i, L):
    """Text and labels for position i: the atom file's own (olens_atoms.py), else the residual run's reads.jsonl row."""
    p = a["positions"][i]
    r = rows.get((a["id"], p, L), {})
    t = {k: v[i] for k, v in a.get("text", {}).items()}
    return {
        "env": a.get("env", r.get("env")), "env_arm": a.get("arm", r.get("env_arm")), "positive": a.get("positive", r.get("positive")),
        "token_index": p, "token": t.get("token", r.get("token")), "role": t.get("role", "assistant"),  # residual reads: assistant only
        "prefix": t.get("prefix", r.get("prefix", "")), "continuation": t.get("continuation", r.get("continuation", "")),
    }  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--extremes", type=int, default=20)
    ap.add_argument("--prefixes", help="comma-separated prefix lengths to keep (default: all stored)")
    args = ap.parse_args()
    rng = random.Random(0)
    files, rows = load(args.run)
    out = []
    for tid, a in files.items():
        lengths = a.get("prefix_lengths", list(range(1, a["cont"][a["layers"][0]].shape[1] + 1)))
        keep = [lengths.index(k) for k in map(int, args.prefixes.split(","))] if args.prefixes else list(range(len(lengths)))
        for L in a["layers"]:
            ok = [i for i, b in enumerate(a["plain_atoms"][L]) if len(b)]
            if not ok:
                continue
            z, C, B = a["z"][L][ok].float(), a["cont"][L][ok][:, keep].float(), [a["plain_atoms"][L][i] for i in ok]
            s = scores(z, C, B)
            nulls = {}
            for name in ("null_same", "null_other"):
                draws = []
                for _ in range(args.draws):
                    draws.append(null_gain(z, C, s["cont"], [draw(rng, files, tid, L, i, name == "null_same") for i in ok]))
                nulls[name] = torch.stack(draws).mean(0)
            c = 0
            for j, i in enumerate(ok):
                m = info(a, rows, i, L)
                ctx = m["prefix"] + m["continuation"]
                bl = [{"text": t, "n_b": round(float(s["n_b"][c + q]), 4), "solo": round(float(s["solo"][c + q]), 4),
                       "overlap": round(overlap(t, ctx), 3)} for q, t in enumerate(a["plain_bullets"][L][i])]  # fmt: skip
                c += len(bl)
                out.append({"id": tid, **m, "layer": L, **{k: round(float(s[k][j]), 4) for k in ("cont", "lens", "joint", "gain")},
                            **{k: round(float(v[j]), 4) for k, v in nulls.items()},
                            "novelty": round(float(s["gain"][j] - nulls["null_other"][j]), 4),
                            "bullets": sorted(bl, key=lambda b: -b["n_b"])})  # fmt: skip
    with open(args.run / "novelty.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    summarize(out)
    extremes(out, args.run / "novelty_extremes.md", args.extremes)


def draw(rng, files, tid, L, i, same):
    """Bullet atoms read elsewhere: same transcript (>= 64 tokens away when possible) or a random other transcript."""
    if same:
        a = files[tid]
        ps, B = a["positions"], a["plain_atoms"][L]
        cand = [j for j in range(len(ps)) if j != i and len(B[j])]
        far = [j for j in cand if abs(ps[j] - ps[i]) >= 64] or cand
        return B[rng.choice(far)] if far else torch.zeros(0)
    others = [t for t in files if t != tid and L in files[t]["layers"]] or [tid]
    B = [b for b in files[rng.choice(others)]["plain_atoms"][L] if len(b)]
    return rng.choice(B) if B else torch.zeros(0)


def rank(x):
    return np.argsort(np.argsort(x))


def flat(s):
    return (s or "").replace("\n", " ⏎ ")


def summarize(out):
    groups = defaultdict(list)
    for r in out:
        groups[(r["layer"], "all")].append(r)
        groups[(r["layer"], r["role"])].append(r)
    for (L, role), rs in sorted(groups.items()):
        col = {k: np.array([r[k] for r in rs]) for k in ("cont", "lens", "joint", "gain", "null_same", "null_other", "novelty")}
        print(f"L{L} {role}: {len(rs)} positions, {len({r['id'] for r in rs})} transcripts")
        for k, v in col.items():
            q10, q90 = np.quantile(v, [0.1, 0.9])
            print(f"  {k:10s} mean {v.mean():+.3f}  median {np.median(v):+.3f}  p10 {q10:+.3f}  p90 {q90:+.3f}")
        g = col["gain"]
        print(f"  gain > null_other {(g > col['null_other']).mean():.0%}, gain > null_same {(g > col['null_same']).mean():.0%}")
        bl = [b for r in rs for b in r["bullets"]]
        if len(bl) < 3:
            continue
        nb, ov, so = (np.array([b[k] for b in bl]) for k in ("n_b", "overlap", "solo"))
        hi = nb[ov >= 0.5].mean() if (ov >= 0.5).any() else float("nan")
        lo = nb[ov == 0].mean() if (ov == 0).any() else float("nan")
        print(f"  {len(bl)} bullets: spearman(n_b, overlap) {np.corrcoef(rank(nb), rank(ov))[0, 1]:+.3f}, "
              f"spearman(solo, overlap) {np.corrcoef(rank(so), rank(ov))[0, 1]:+.3f}; mean n_b at overlap >= 0.5 {hi:.4f} "
              f"vs overlap 0 {lo:.4f}")  # fmt: skip


def extremes(out, path, n):
    md = ["# Novelty extremes\n", "novelty = FVE a read adds over the realized continuation, minus another transcript's bullets' gain.\n"]
    by_L = defaultdict(list)
    for r in out:
        by_L[r["layer"]].append(r)
    for L, rs in sorted(by_L.items()):
        per = defaultdict(list)
        for r in rs:
            per[r["id"]].append(r["novelty"])
        md.append(f"\n## L{L}: transcripts by mean novelty\n")
        md += [f"- {np.mean(v):+.4f} `{t}` ({len(v)} positions)" for t, v in sorted(per.items(), key=lambda kv: -np.mean(kv[1]))]
        rs = sorted(rs, key=lambda r: -r["novelty"])
        for title, sel in (("highest", rs[:n]), ("lowest", rs[::-1][:n])):
            md.append(f"\n## L{L}: {n} {title}-novelty positions\n")
            for r in sel:
                head = f"{r['novelty']:+.4f} `{r['id']}` @ {r['token_index']}"
                md.append(f"### {head} ({r['role']}, {r['env_arm']}, positive={r['positive']})\n")
                fves = " · ".join(f"{k} {r[k]:.3f}" for k in ("cont", "lens", "joint", "gain", "null_other"))
                md.append(fves + "\n")
                md.append(f"> …{flat(r['prefix'])} **‖** {flat(r['continuation'])}\n")
                md += [f"- n_b {b['n_b']:+.4f} · solo {b['solo']:.3f} · overlap {b['overlap']:.2f}: {b['text']}" for b in r["bullets"]]
                md.append("")
    path.write_text("\n".join(md))
    print(f"wrote {path.parent / 'novelty.jsonl'}, {path}")


if __name__ == "__main__":
    main()
