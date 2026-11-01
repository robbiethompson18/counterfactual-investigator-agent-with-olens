"""Plan the natural-rate blog-author sample from the pairs already deleted (Kylie, 2026-10-31: "do or simulate a natural
sample", "avoid repeated work by sampling from our pool of already computed ones").

Cells are sex x age band x question x Qwen right/wrong. A cell's natural count of confident pairs is its eligible-corpus
bloggers (every blogger blog.py build would accept, used or not) times the cell's confident-pair rate among the bloggers
answered in batches 1-3; the batches stratified only by sex x age band, so within a cell every answered post is a uniform
draw. Each cell's target is scale x natural, the scale set so the expected case count (each cell's mean stable-with-positive
cases per deleted pair) is ~1,600; a random subsample of the cell's deleted pairs fills it, and any shortfall is a uniform
draw from the cell's confident, answered, never-deleted pairs (the new deletions, both option orders). Writes RUN/plan.json
(the cell table), RUN/pairs.jsonl, RUN/new_pairs.txt and RUN/inputs_del_answer.jsonl; then run the deletions and
`blog_u cases RUN RUN/del data/runs/author_mcq_blog/2026-10-31_blog-c/del`.

    uv run --with numpy --with transformers --with jinja2 python -m scripts.author_probe.natural_plan RUN [--cases 1600]
"""

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np

from scripts.author_probe.bank import MODEL, orders
from scripts.author_probe.blog import ATTRS, BANK, _row, conf, join
from scripts.author_probe.blog_u import OLD_INPUTS, load, logps

R = Path("data/runs/author_mcq_blog")
BATCH3_SHARDS = R / "2026-10-31_blog-c/del"  # batch 3's one-order shards (the rest of its deletions)
# eligible bloggers per (sex, age band) in the whole corpus: the bank's batches 1-4 plus the ~50 build left over
ELIGIBLE = {("female", "<20"): 3386, ("female", "20s"): 3305, ("female", "30+"): 1142,
            ("male", "<20"): 3108, ("male", "20s"): 3029, ("male", "30+"): 1005}  # fmt: skip


def band(age: int) -> str:
    return "<20" if age < 20 else "20s" if age < 30 else "30+"


def main(a):
    run = Path(a.run)
    passages = {p["id"]: p for p in load(BANK / "passages.jsonl")}
    cell = lambda pid: (passages[pid]["profile"]["gender"], band(passages[pid]["profile"]["age"]))
    ans = [r for b in ("", "-b", "-c") for r in load(R / f"2026-10-30_qwen36-qfirst{b}/qwen_answers.jsonl")]
    posts = collections.Counter(cell(p) for p in {r["passage"] for r in ans})
    key = lambda r: cell(r["passage"]) + (r["attr"], r["answer"] != r["label"])
    confident = [r for r in ans if conf(r["pmf"]) >= 0.5 and r["order_agree"]]
    rate = collections.Counter(key(r) for r in confident)
    natural = {k: ELIGIBLE[k[:2]] * v / posts[k[:2]] for k, v in rate.items()}
    order_of = {r["id"]: r["order"] for f in OLD_INPUTS for r in load(f)}
    lp = logps([str(BATCH3_SHARDS)])

    def pmf(k):
        ps = []
        for o in (0, 1):
            if (i := f"{k}:o{o}") in lp:
                x = np.exp(np.array(lp[i]) - max(lp[i]))
                c = np.zeros(len(x))
                c[order_of[i]] = x / x.sum()
                ps.append(c)
        return sum(ps) / len(ps) if ps else None

    amap = {r["id"]: r for r in ans}
    pool = []
    for b in ("", "-b", "-c"):
        for pr in load(R / f"2026-10-30_ablate-loo{b}/pairs.jsonl"):
            full, dels = pmf(f"{pr['id']}~-1"), [pmf(f"{pr['id']}~{i}") for i in range(pr["n"])]
            if full is None or any(d is None for d in dels):
                continue
            ans_i = int(full.argmax())
            drops = np.array([full[ans_i] - d[ans_i] for d in dels])
            npos = int((drops > 0.15).sum())
            stable = np.median([0.5 * np.abs(d - full).sum() for d in dels]) < 0.05
            pool.append({"id": pr["id"], "key": key(amap[pr["id"]]), "cases": (npos + min(npos, len(drops) - npos)) * stable})
    deleted = {p["id"] for p in pool}
    avail = collections.defaultdict(list)
    for r in confident:
        if r["id"] not in deleted:
            avail[key(r)].append(r["id"])
    by_key = collections.defaultdict(list)
    for p in pool:
        by_key[p["key"]].append(p)
    exp = {k: np.mean([p["cases"] for p in v]) for k, v in by_key.items()}
    scale = a.cases / sum(natural[k] * exp[k] for k in by_key)
    rng = random.Random(1)
    table, take, new = [], [], []
    for k in sorted(natural):
        target = round(scale * natural[k])
        have = [p["id"] for p in by_key[k]]
        old = rng.sample(have, min(target, len(have)))
        fresh = rng.sample(sorted(avail[k]), min(max(0, target - len(have)), len(avail[k])))
        take += old
        new += fresh
        table.append({"cell": [*k[:3], "wrong" if k[3] else "right"], "natural_pairs": round(natural[k]), "deleted_pairs": len(have),
                      "target": target, "from_existing": len(old), "new_deletions": len(fresh)})  # fmt: skip
    run.mkdir(parents=True, exist_ok=True)
    (run / "plan.json").write_text(json.dumps({"seed": 1, "scale": scale, "target_cases": a.cases, "cells": table}, indent=1))
    pairs = [{"id": i, "passage": i.split(":")[0], "attr": i.split(":")[1], "n": len(passages[i.split(":")[0]]["sentences"])}
             for i in take + new]  # fmt: skip
    (run / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    (run / "new_pairs.txt").write_text("\n".join(new))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    with open(run / "inputs_del_answer.jsonl", "w") as f:
        for pair in new:
            pid, attr = pair.split(":")
            sents = passages[pid]["sentences"]
            for i in range(-1, len(sents)):
                text = join([s for j, s in enumerate(sents) if j != i])
                rows = (_row(tok, f"{pair}~{i}:o{o}", text, attr, order) for o, order in enumerate(orders(len(ATTRS[attr]))))
                f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"scale {scale:.4f}: {len(take)} deleted pairs reused, {len(new)} new pairs to delete (both orders)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cases", type=int, default=1600)
    main(ap.parse_args())
