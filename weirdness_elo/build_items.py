"""Build run/items.json (judge-facing) and run/transcripts/<id>.json (dashboard-only) from the 9 CHIVE eval runs.

Verified = verification score >= 7 (CHIVE's min_verification_score). Stratified sample of N_SAMPLE across runs
(proportional allocation) if there are more verified investigations than that."""

import json
import random
import re
from pathlib import Path

D = Path("data/datasets")
OUT = Path("run")
MIN_VER = 7
N_SAMPLE = 1500
RUNS = sorted(p.name for p in D.iterdir() if p.name.endswith("_eval"))


def short(s, max_sent=3, max_chars=450):
    s = re.sub(r"\s+", " ", s or "").strip()
    sents = re.split(r"(?<=[.!?])\s+", s)
    out = " ".join(sents[:max_sent])
    return out if len(out) <= max_chars else out[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def jl(p):
    with open(p) as fh:
        for l in fh:
            if l.strip():
                yield json.loads(l)


ver = {r: {v["prompt_id"]: v["score"] for v in jl(D / r / "verification.jsonl")} for r in RUNS}
counts = {}
cands = []
for r in RUNS:
    n_inv = n_ok = 0
    for inv in jl(D / r / "investigations.jsonl"):
        n_inv += 1
        s = ver[r].get(inv["prompt_id"])
        if s is None or s < MIN_VER or not inv.get("question"):
            continue
        n_ok += 1
        cands.append((r, inv["prompt_id"], s))
    counts[r] = {"investigations": n_inv, "verified": n_ok}
tot = sum(c["verified"] for c in counts.values())
rng = random.Random(0)
by_run = {r: [c for c in cands if c[0] == r] for r in RUNS}
keep = set()
if tot > N_SAMPLE:
    alloc = {r: round(N_SAMPLE * len(v) / tot) for r, v in by_run.items()}
    for r, v in by_run.items():
        for c in rng.sample(v, min(alloc[r], len(v))):
            keep.add((c[0], c[1]))
else:
    keep = {(c[0], c[1]) for c in cands}
for r in RUNS:
    counts[r]["sampled"] = sum(1 for k in keep if k[0] == r)
print(json.dumps(counts, indent=1), "total verified", tot, "sampled", len(keep))

OUT.mkdir(exist_ok=True)
(OUT / "transcripts").mkdir(exist_ok=True)
items = []
for r in RUNS:
    want = {pid for rr, pid in keep if rr == r}
    comp = {c["id"]: c for c in jl(D / r / "completions.jsonl") if c["id"] in want}
    for inv in jl(D / r / "investigations.jsonl"):
        pid = inv["prompt_id"]
        if pid not in want:
            continue
        sf = inv.get("structured_findings") or {}
        iid = f"{r}:{pid}"
        beh = sf.get("baseline_behavior") or inv.get("behavior_summary")
        items.append(
            {
                "id": iid,
                "run": r,
                "prompt_id": pid,
                "question": inv["question"],
                "behavior": short(beh),
                "verification_score": ver[r][pid],
                "interest_score": inv.get("interest_score"),
                "baseline_rate": f"{inv.get('baseline_grader_yes')}/{len(inv.get('completions') or [])}",
            }
        )
        c = comp.get(pid, {})
        msgs = list(inv.get("messages") or c.get("messages") or [])
        (OUT / "transcripts" / f"{iid.replace(':', '__')}.json").write_text(
            json.dumps(
                {
                    "id": iid,
                    "run": r,
                    "prompt_id": pid,
                    "question": inv["question"],
                    "behavior_full": inv.get("behavior_summary"),
                    "baseline_behavior": sf.get("baseline_behavior"),
                    "classifier_question": inv.get("classifier_question"),
                    "messages": msgs,
                    "user_message": inv.get("user_message"),
                    "completions": inv.get("completions") or c.get("completions") or [],
                    "grader_verdicts": inv.get("baseline_grader_verdicts"),
                    "explanation": sf.get("structured_answer") or inv.get("investigation_report"),
                    "commentary": sf.get("commentary"),
                    "confidence": sf.get("confidence"),
                    "experiments": sf.get("experiment_summaries"),
                    "verification_score": ver[r][pid],
                }
            )
        )
rng.shuffle(items)
(OUT / "items.json").write_text(json.dumps(items, indent=1))
print("items", len(items))
