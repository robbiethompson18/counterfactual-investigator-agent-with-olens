"""O-lens reads for a few combo_regions cells.

    header:     greedy reads at the 4 non-newline header tokens of the final user turn x LAYERS, for the direct,
                cloud and auth_header prompts of each cell.
    preanswer:  for sampled auth_header (or other probe) replies, the tokens from `</think>` up to the first answer
                token x LAYERS.

    uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/combo_regions/lens.py \
        header A:aws_usw2 A:ali_hz ...
    ... lens.py preanswer auth_header 6 A:aws_usw2 ...
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/combo_regions")
from cues import CLOUD, CREATOR, PROBES, first, msgs
from qwen import header_positions, read, render

OUT = Path("data/runs/identity/2026-10-21_combo_regions")
LAYERS = (32, 40, 48, 56)


def label(text):
    return {"creator": first(text, CREATOR), "cloud": first(text, CLOUD)}


def write(path, meta, items, res, toks):
    with path.open("a") as f:
        for (p, L), r in zip(items, res):
            for k, b in enumerate(r[0]["bullets"]):
                bf = r[0]["bfve"][k] if k < len(r[0]["bfve"]) else None
                f.write(
                    json.dumps(
                        {**meta, "pos": p, "tok": toks[p], "layer": L, "b": k, "bullet": b, "bfve": bf, **label(b)}, ensure_ascii=False
                    )
                    + "\n"
                )


def header(cells, probes=("direct", "cloud", "auth_header")):
    path = OUT / "lens_header.jsonl"
    done = {(json.loads(x)["cell"], json.loads(x)["probe"]) for x in path.open()} if path.exists() else set()
    for c in cells:
        b, reg = c.split(":")
        for p in probes:
            if (c, p) in done:
                continue
            ids, toks = render(msgs(b, reg, PROBES[p]))
            items = [(q, L) for q in header_positions(ids) if toks[q] != "\n" for L in LAYERS]
            res = read(ids, items)
            write(path, {"cell": c, "probe": p}, items, res, toks)
            print(c, p, len(items), flush=True)


def preanswer(probe, per_cell, cells, seed=0):
    rows = [json.loads(x) for x in (OUT / "samples.jsonl").open()]
    path = OUT / "lens_preanswer.jsonl"
    rng = random.Random(seed)
    for c in cells:
        b, reg = c.split(":")
        rs = [r for r in rows if r["base"] == b and r["region"] == reg and r["probe"] == probe and r["label"] != "trunc"]
        rng.shuffle(rs)
        # balance labels: round-robin over labels so a minority answer gets read too
        by = {}
        for r in rs:
            by.setdefault(r["label"], []).append(r)
        pick = []
        while len(pick) < per_cell and any(by.values()):
            for lab in list(by):
                if by[lab] and len(pick) < per_cell:
                    pick.append(by[lab].pop())
        for r in pick:
            ids, toks = render(msgs(b, reg, PROBES[probe]), r)
            t = max(i for i, s in enumerate(toks) if "</think>" in s)
            ans = next(i for i in range(t + 1, len(toks)) if toks[i].strip())
            items = [(q, L) for q in range(t, ans) for L in LAYERS]
            res = read(ids, items)
            write(path, {"cell": c, "probe": probe, "sample_i": r["i"], "answer_label": r["label"]}, items, res, toks)
            print(c, probe, r["i"], r["label"], len(items), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "header":
        header(sys.argv[2:])
    else:
        preanswer(sys.argv[2], int(sys.argv[3]), sys.argv[4:])
