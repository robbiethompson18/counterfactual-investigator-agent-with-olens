import collections
import glob
import json
import os
import sys
from pathlib import Path

S = sys.argv[1]
H = [json.loads(l) for l in Path(f"{S}/hits3.jsonl").read_text().splitlines()]
# arm names from transcript records
want = {h["id"].split("|")[0] for h in H}
arm = {}
roots = [os.path.dirname(f) for f in Path(f"{S}/files.txt").read_text().split()] + glob.glob(
    "/home/claude/workspace/tmp/qwen_hf_reads/runs/olens_reads/*"
)
for r in set(roots):
    for c in glob.glob(f"{r}/requests*.jsonl") + glob.glob(f"{r}/*transcripts.jsonl"):
        for l in Path(c).read_text().splitlines():
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                continue
            if d.get("id") in want and d.get("arm") is not None:
                arm.setdefault(d["id"], str(d["arm"]))
ENV = {"qwen": "donation_bet (thinking off)", "prefix": "character_drift", "reply": "character_drift"}
J = {}
for b in [d for d in glob.glob(f"{S}/triage/[bc][0-9]") if os.path.isdir(d)]:
    ins = [json.loads(l) for l in Path(f"{b}/input.jsonl").read_text().splitlines()]
    outs = {json.loads(l)["key"]: json.loads(l) for l in Path(f"{b}/output.jsonl").read_text().splitlines()}
    for i in ins:
        J[i["key"]] = (os.path.basename(b)[0], outs[i["key"]])
rows: list[dict] = []
for k, (rnd, o) in J.items():
    h = H[k]
    a = arm.get(h["id"].split("|")[0]) or h["arm"]
    rows.append(
        {
            "key": k,
            "round": rnd,
            "env": ENV.get(h["env"], h["env"]),
            "arm": a,
            "id": h["id"],
            "label": o["label"],
            "s": o.get("strength") or 0,
            "text": h["text"],
            "role": h["role"],
            "layer": h["layer"],
        }
    )
Path(f"{S}/final_rows.json").write_text(json.dumps(rows))
Path(f"{S}/arms.json").write_text(json.dumps(arm))
# env table from the 30/env sample
G = json.loads(Path(f"{S}/groups2.json").read_text())
pos = collections.Counter()
eps = collections.Counter()
for g in G:
    e = ENV.get(g["env"], g["env"])
    pos[e] += g["positions"]
    eps[e] += g["episodes"]
sil = collections.Counter(ENV.get(h["env"], h["env"]) for h in H if not h["echo"] and not h["verbal"])
smp: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
for r in rows:
    if r["round"] == "b":
        smp[r["env"]]["n"] += 1
        smp[r["env"]][r["label"]] += 1
        smp[r["env"]]["s2"] += r["label"] == "eval_aware" and int(r["s"]) >= 2
print("| env | episodes read | distinct positions | silent regex bullets | sampled | eval-aware | strength ≥2 | est. strength ≥2 bullets |")
print("|---|---|---|---|---|---|---|---|")
for e in sorted(smp, key=lambda e: -sil[e] * smp[e]["s2"] / smp[e]["n"]):
    n = smp[e]["n"]
    print(f"| {e} | {eps[e]} | {pos[e]} | {sil[e]} | {n} | {smp[e]['eval_aware']} | {smp[e]['s2']} | {round(sil[e] * smp[e]['s2'] / n)} |")
