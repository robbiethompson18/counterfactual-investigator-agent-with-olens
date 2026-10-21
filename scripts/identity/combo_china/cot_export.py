"""Export blinded, excerpted censorship-probe CoTs of the given cues for Haiku labelling (cot_rubric.md).
usage: python cot_export.py cue1,cue2,... [k_end] [tag] [k_start] -> cot_grade/batch_XX.jsonl (+ key.json)"""

import json
import random
import re
import sys
from pathlib import Path

D = Path("data/runs/identity/2026-10-21_combo_china")
KW = re.compile(
    r"\bI am\b|\bI'm\b|as an AI|\bmy (?:instructions|guidelines|developer|creator|system)|system prompt|\bmodel\b|law|regulat|region|deploy|locale|PRC|official|national|中国|法|境内|立场|identity|operator|user is",
    re.IGNORECASE,
)


def excerpt(r, head=3000, tail=1500, maxkw=12):
    if len(r) <= head + tail + 500:
        return r
    mid = r[head:-tail]
    lines = [ln.strip() for ln in mid.split("\n") if KW.search(ln) and len(ln.strip()) > 10][:maxkw]
    return r[:head] + "\n[...]\n" + "\n[...]\n".join(ln[:400] for ln in lines) + "\n[...]\n" + r[-tail:]


def main():
    cues = sys.argv[1].split(",")
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    tag = sys.argv[3] if len(sys.argv) > 3 else "a"
    kmin = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    items = []
    for line in (D / "samples.jsonl").open():
        d = json.loads(line)
        if d["cue"] in cues and d["probe"] in ("xinjiang", "tiananmen", "xi") and kmin <= d["k"] < per:
            items.append(d)
    random.Random(0).shuffle(items)
    out = D / "cot_grade"
    keyf = out / f"key_{tag}.json"
    key = {}
    batches = [items[i : i + 60] for i in range(0, len(items), 60)]
    for b, chunk in enumerate(batches):
        with (out / f"batch_{tag}{b:02d}.jsonl").open("w") as f:
            for d in chunk:
                cid = f"{tag}{len(key):04d}"
                key[cid] = [d["cue"], d["probe"], d["k"]]
                q = {"xinjiang": "Uyghurs in Xinjiang", "tiananmen": "Tiananmen Square June 1989", "xi": "criticisms of Xi Jinping"}[
                    d["probe"]
                ]
                f.write(json.dumps({"id": cid, "question_topic": q, "reasoning": excerpt(d["reasoning"] or "")}, ensure_ascii=False) + "\n")
    json.dump(key, keyf.open("w"))
    print(len(items), "items in", len(batches), "batches")


if __name__ == "__main__":
    main()
