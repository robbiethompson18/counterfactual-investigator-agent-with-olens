"""Merge Haiku CoT labels (cot_grade/labels_*.jsonl) back to (cue, probe) cells; print per-cue rates and agreement with regex."""

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts/identity/combo_china")
from analyze import COT_CNREG, COT_CUE, COT_SELF

D = Path("data/runs/identity/2026-10-21_combo_china")


def labels():
    key = {}
    for f in glob.glob(str(D / "cot_grade/key_*.json")):
        key |= json.load(open(f))
    out = {}
    for f in sorted(glob.glob(str(D / "cot_grade/labels_*.jsonl"))):
        for line in open(f):
            if line.strip():
                d = json.loads(line)
                if d["id"] in key:
                    out[tuple(key[d["id"]])] = d
    return out


def main():
    lab = labels()
    S = {(d["cue"], d["probe"], d["k"]): d for d in map(json.loads, (D / "samples.jsonl").open())}
    per = defaultdict(lambda: defaultdict(int))
    agree = defaultdict(lambda: [0, 0])
    for k, v in lab.items():
        c = per[k[0]]
        c["n"] += 1
        for f in ("self_identity", "own_jurisdiction", "cue_ref"):
            c[f] += bool(v.get(f))
        r = S[k]["reasoning"] or ""
        for f, rx in (("self_identity", COT_SELF), ("own_jurisdiction", COT_CNREG), ("cue_ref", COT_CUE)):
            agree[f][0] += bool(v.get(f)) == bool(rx.search(r))
            agree[f][1] += 1
    print("cue\tn\tself_identity\town_jurisdiction\tcue_ref")
    for cue, c in per.items():
        print(cue, c["n"], c["self_identity"], c["own_jurisdiction"], c["cue_ref"], sep="\t")
    print("regex-vs-haiku agreement:", {f: f"{a}/{b}" for f, (a, b) in agree.items()})
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in lab.items()}, open(D / "cot_labels.json", "w"))


if __name__ == "__main__":
    main()
