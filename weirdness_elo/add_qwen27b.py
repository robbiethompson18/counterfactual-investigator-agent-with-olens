"""Qwen3.6-27B CHIVE behaviours -> run/items_extra.json (dashboard only, unranked; not in items.json so elo.py ignores them).

Sources (read-only):
  - agu18dec/chive-forensics (HF): 300 Qwen3.6-27B incidents, each with supported causes (all count as verified)
  - weirdchat-chive CHIVE runs wc30_* on disk: same schema as chive-data; keep verification score >= 7"""

import json
import os
import re
from pathlib import Path

MIN_VER = 7


def short(s, max_sent=3, max_chars=450):  # same as build_items.short
    s = re.sub(r"\s+", " ", s or "").strip()
    sents = re.split(r"(?<=[.!?])\s+", s)
    out = " ".join(sents[:max_sent])
    return out if len(out) <= max_chars else out[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def jl(p):
    with open(p) as fh:
        for l in fh:
            if l.strip():
                yield json.loads(l)


OUT = Path("run")
WC = Path(
    os.environ.get("WEIRDCHAT_CHIVE_RUNS", os.path.expanduser("~/workspace/weirdchat-chive/weirdchat-chive/data/runs"))
)  # CHIVE runs on WeirdChat (weirdchat-chive branch)
F = Path("data/forensics/data")
extra = []


def save(iid, d):
    (OUT / "transcripts" / f"{iid.replace(':', '__')}.json").write_text(json.dumps(d))


# 1. Agam's forensics set
gt = {g["prompt_id"]: g for g in jl(F / "ground_truth.jsonl")}
for inc in jl(F / "incidents.jsonl"):
    pid = inc["prompt_id"]
    g = gt[pid]
    run = "qwen3p6_27b_chive_forensics"
    iid = f"{run}:{pid}"
    causes = "\n\n".join(
        f"{t['id']} ({t.get('status')}): {t['cause']}\n" + "\n".join(f"  - {e}" for e in t.get("evidence", [])) for t in g["targets"]
    )
    ruled = "\n\n".join(
        f"{x['id']} ({x.get('kind')}): {x['cause']}\n" + "\n".join(f"  - {e}" for e in x.get("evidence", []))
        if "cause" in x
        else f"{x['id']} ({x.get('kind')}, {x.get('effect_pp')} pp): {x['edit']}"
        for x in g["ruled_out"]
    )
    extra.append(
        {
            "id": iid,
            "run": run,
            "prompt_id": pid,
            "question": inc["behaviour"],
            "behavior": short(g["observed"]),
            "verification_score": None,
            "baseline_rate": f"{inc['k']}/{inc['n']}",
        }
    )
    save(
        iid,
        {
            "id": iid,
            "run": run,
            "prompt_id": pid,
            "question": inc["behaviour"],
            "behavior_full": g["observed"],
            "classifier_question": inc["behaviour"],
            "messages": inc["messages"],
            "completions": [inc["completion"]],
            "grader_verdicts": None,
            "explanation": f"CAUSES\n\n{causes}\n\nRULED OUT\n\n{ruled}",
            "experiments": g.get("experiments"),
            "verification_score": None,
        },
    )

# 2. weirdchat-chive wc30 runs (chive schema)
for rd in sorted(WC.glob("wc30_*")):
    if not (rd / "investigations.jsonl").exists() or not (rd / "verification.jsonl").exists():
        continue
    ver = {v["prompt_id"]: v["score"] for v in jl(rd / "verification.jsonl")}
    for inv in jl(rd / "investigations.jsonl"):
        pid = inv["prompt_id"]
        s = ver.get(pid)
        if s is None or s < MIN_VER or not inv.get("question"):
            continue
        run = f"qwen3p6_27b_{rd.name}"
        iid = f"{run}:{pid}"
        sf = inv.get("structured_findings") or {}
        extra.append(
            {
                "id": iid,
                "run": run,
                "prompt_id": pid,
                "question": inv["question"],
                "behavior": short(sf.get("baseline_behavior") or inv.get("behavior_summary")),
                "verification_score": s,
                "baseline_rate": f"{inv.get('baseline_grader_yes')}/{len(inv.get('completions') or [])}",
            }
        )
        save(
            iid,
            {
                "id": iid,
                "run": run,
                "prompt_id": pid,
                "question": inv["question"],
                "behavior_full": inv.get("behavior_summary"),
                "baseline_behavior": sf.get("baseline_behavior"),
                "classifier_question": inv.get("classifier_question"),
                "messages": inv.get("messages") or [],
                "completions": inv.get("completions") or [],
                "grader_verdicts": inv.get("baseline_grader_verdicts"),
                "explanation": sf.get("structured_answer") or inv.get("investigation_report"),
                "commentary": sf.get("commentary"),
                "confidence": sf.get("confidence"),
                "experiments": sf.get("experiment_summaries"),
                "verification_score": s,
            },
        )

(OUT / "items_extra.json").write_text(json.dumps(extra, indent=1))
from collections import Counter

print(Counter(e["run"] for e in extra))
