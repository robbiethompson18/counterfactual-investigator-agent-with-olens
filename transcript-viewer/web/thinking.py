"""Thinking-on resamples: Qwen3.6-27B's reply to a favourite WeirdChat transcript's (docs/favorite-transcripts.md) user
turn, sampled again with thinking on. One viewer pattern per original transcript, in the WeirdChat tab under the
behavior "thinking-resample".

Source: data/thinking.jsonl (other machines' copies come through scripts/publish.py like the reads), written by
scripts/thinking.py. Each record holds the exact token ids: the conversation up to the original reply, templated with
thinking on, then the generated thinking and reply. Reads are ordinary base-Qwen reads, cached in data/reads.jsonl under
the resample's tid.
"""

from organisms import msg_ranges

BEHAVIOR = "thinking-resample"


def load(S, records):
    """Append one pattern per original transcript to S["patterns"], and its resamples to S["trs"]."""
    tok = S["tok"]
    start, think, end = (tok.convert_tokens_to_ids(x) for x in ("<|im_start|>", "<think>", "<|im_end|>"))
    pats = {p["pattern_id"]: p for p in S["patterns"]}
    by_parent = {}
    for r in records:  # later shards win
        by_parent.setdefault(r["parent"], {})[r["tid"]] = r
    for parent, rs in by_parent.items():
        t0 = S["trs"][parent]
        p0 = pats[t0["pattern_id"]]
        rs = sorted(rs.values(), key=lambda r: ("source" in r, r["sample"]))  # our vLLM samples, then OpenRouter's
        for r in rs:
            # earlier messages as usual; the reply's content starts at its thinking (which may hit the cap with no </think>)
            ids = r["ids"]
            last = max(i for i, x in enumerate(ids) if x == start)
            a = ids.index(think, last) + 1
            ranges = [*msg_ranges(tok, ids[:last]), [a, ids.index(end, a) if end in ids[a:] else len(ids)]]
            S["trs"][r["tid"]] = {
                "transcript_id": r["tid"],
                "pattern_id": f"thinking:{parent}",
                "prompt_id": "thinking",  # matches no Claude rollouts
                "messages": [*t0["messages"][:-1], {"role": "assistant", "content": r["reply"]}],
                "ids": r["ids"],
                "msg_tok": ranges,
                "judgment": {"match": False, "explanation": ""},
                "is_highlight": r is rs[0],
                "weirdchat_url": t0["weirdchat_url"],
                "thinking": {
                    "source": r.get("source", "vLLM, bf16"),
                    **{k: r[k] for k in ("parent", "sample", "temperature", "n_new", "hit_cap")},
                },
            }
        S["patterns"].append(
            {
                "idx": len(S["patterns"]),
                "ds": "weirdchat",
                "pattern_id": f"thinking:{parent}",
                "title": f"Thinking on: {p0['title']}",
                "behavior_id": BEHAVIOR,
                "method": "resample with thinking on",
                "metrics": {},
                "n_prompts": 1,
                "tids": [r["tid"] for r in rs],
                "weirdchat_url": t0["weirdchat_url"],
                "elo": {},
                "description": p0["description"],
                "representative_user_text": t0["messages"][0]["content"],
                "cluster_label": p0["cluster_label"],
                "openrouter_replication": None,
                "thinking": {"parent": parent, "parent_idx": p0["idx"]},
            }
        )
