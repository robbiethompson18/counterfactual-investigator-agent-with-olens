"""Confession follow-ups: a favourite WeirdChat transcript (docs/favorite-transcripts.md) plus one more user turn asking
Qwen why it did what it did, and Qwen3.6-27B's answer. One viewer pattern per original transcript, in the WeirdChat tab
under the behavior "confession-follow-up".

Source: data/followups.jsonl (other machines' copies come through scripts/publish.py like the reads), written by
scripts/followups.py. Each record holds the exact token ids: the conversation re-templated with the new user turn (so,
as Qwen was trained, the earlier reply loses its empty think block), then the generated answer. Reads are ordinary
base-Qwen reads, cached in data/reads.jsonl under the follow-up's tid.
"""

from organisms import msg_ranges

BEHAVIOR = "confession-follow-up"
WORDINGS = ("named", "neutral")  # scripts/followups.py: names the weird behavior / asks only "why did you respond that way?"


def load(S, records):
    """Append one pattern per original transcript to S["patterns"], and its follow-ups to S["trs"]."""
    tok = S["tok"]
    pats = {p["pattern_id"]: p for p in S["patterns"]}
    by_parent = {}
    for r in records:  # later shards win
        by_parent.setdefault(r["parent"], {})[r["tid"]] = r
    for parent, rs in by_parent.items():
        t0 = S["trs"][parent]
        p0 = pats[t0["pattern_id"]]
        rs = sorted(rs.values(), key=lambda r: (WORDINGS.index(r["wording"]), r["sample"]))
        for r in rs:
            S["trs"][r["tid"]] = {
                "transcript_id": r["tid"],
                "pattern_id": f"followup:{parent}",
                "prompt_id": f"followup:{r['wording']}",  # matches no Claude rollouts
                "messages": [
                    *t0["messages"],
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": r["reply"]},
                ],
                "ids": r["ids"],
                "msg_tok": msg_ranges(tok, r["ids"]),
                "judgment": {"match": False, "explanation": ""},
                "is_highlight": r["sample"] == 0,
                "weirdchat_url": t0["weirdchat_url"],
                "followup": {k: r[k] for k in ("parent", "wording", "question", "sample", "temperature", "n_new", "hit_cap")},
            }
        S["patterns"].append(
            {
                "idx": len(S["patterns"]),
                "ds": "weirdchat",
                "pattern_id": f"followup:{parent}",
                "title": f"Follow-up ({rs[0].get('label', 'favourite')}): {p0['title']}",
                "behavior_id": BEHAVIOR,
                "method": "follow-up user turn",
                "metrics": {},
                "n_prompts": len({r["wording"] for r in rs}),
                "tids": [r["tid"] for r in rs],
                "weirdchat_url": t0["weirdchat_url"],
                "elo": {},
                "description": p0["description"],
                "representative_user_text": t0["messages"][0]["content"],
                "cluster_label": p0["cluster_label"],
                "openrouter_replication": None,
                "followup": {"parent": parent, "parent_idx": p0["idx"], "questions": {r["wording"]: r["question"] for r in rs}},
            }
        )
