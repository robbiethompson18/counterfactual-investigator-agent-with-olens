"""Render a coding run's rows.jsonl the way Celeste's GLM-5.3 olens dataset does and list every position to read.

Positions = every punctuation / newline token (Celeste's is_punct) PLUS every chat-template / special token (ids >= 154820),
index >= 2 (never [gMASK]<sop>), < 32,768 tokens; episodes longer than that also get Celeste's approximate `late` view
(head through the first user message + "[k earlier messages omitted for length]" + the longest tail that fits; positions only in the tail).
A position whose exact token prefix was already queued is not read again (dense_prefix_dups.parquet points at the canonical one).

    python prep.py --rows rows.jsonl --run-dir runs/coding/<run> --realism <checkout>/donehack/realism \
        --tok <GLM-5.3 tokenizer dir> --out prep/<run>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import render as R

CAP = 32768
FIRST_ADDED = 154820
MARKERS = ("<|system|>", "<|user|>", "<|assistant|>", "<|observation|>")


def is_punct(piece: str) -> bool:
    s = piece.strip()
    return "\n" in piece or (bool(s) and not any(c.isalnum() for c in s))


def specs_for(rows, realism: str):
    if "tools" in rows[0]:  # rows that carry their own tool schemas (overeager_rows.py)
        return [r["tools"] for r in rows]
    if str(rows[0].get("env", "")).startswith("donation_bet") or rows[0].get("env") == "alignment_faking":  # single turn, no tools
        return [None] * len(rows)
    if rows[0].get("env") in ("funding_email_real", "email_ambiguous_real"):  # evalcue harness: `realism` = the evalcue checkout root
        import importlib

        sys.path.insert(0, realism)
        from evalcue import harness

        E = importlib.import_module(f"evalcue.envs.{rows[0]['env']}")
        harness.MODEL = rows[0]["model"]  # the system prompt names the model it was run with
        out = []
        for r in rows:
            ep = E.build(r["arm"], r["item"])
            assert r["messages"][0] == ep.messages[0], (r["arm"], "system prompt differs from evalcue's build()")
            out.append(ep.tools)
        return out
    sys.path.insert(0, realism)
    import realvariants as V  # loads run.py as sys.modules['realrun']

    out = []
    for r in rows:
        if r.get("harness") == "realism_email":
            import email_env

            variant, _, level = r["arm"].partition("@")  # exported arms carry a run tag after the level: base@lowc_junebug
            arm = f"{variant}@{level.split('_')[0]}" if level else variant
            sp = email_env.build(arm, int(r["item"].rsplit("_", 1)[1]), int(r["seed"]))[0]
        else:
            sp = V.VARIANTS[r["arm"]](sys.modules["realrun"].load_problem(r["item"]), int(r["seed"]))
        m0 = r["messages"][0]
        assert (m0["role"] == "system" and m0["content"] == sp.system) or (not sp.system and m0["role"] != "system"), (r["arm"], r["item"])
        out.append(sp.tools)
    return out


def labels_json(r: dict, extra: dict) -> str:
    d = {}
    for k, v in r.items():
        if k in ("messages", "files_before", "files_after", "tools"):
            continue
        d[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    return json.dumps(d | extra)


RAW_PREFIX, RAW_EMPTY_THINK = "[gMASK]<sop><|system|>Reasoning Effort: Max", "<|assistant|><think></think>"


def raw_render(msgs) -> str:
    """evalcue EVALCUE_RAW_PREFILL render (single turn, no tools): GLM prefix + turns + an empty think block + the reply, verbatim."""
    turn = {"system": "<|system|>", "user": "<|user|>"}
    head = RAW_PREFIX + "".join(turn[m["role"]] + m["content"] for m in msgs[:-1])
    assert msgs[-1]["role"] == "assistant" and not msgs[-1].get("reasoning_content")
    return head + RAW_EMPTY_THINK + (msgs[-1]["content"] or "")


def spans(msgs, tools, text, raw=False):
    """char start of each message in the render (prefix property of the GLM template), plus the end."""
    if raw:
        starts, k = [], len(RAW_PREFIX)
        for m in msgs[:-1]:
            starts.append(k)
            k += len({"system": "<|system|>", "user": "<|user|>"}[m["role"]]) + len(m["content"])
        return starts + [k, len(text)]
    starts = []
    for i in range(len(msgs)):
        s = R.render(msgs[:i], tools) if i else R.render([{"role": "user", "content": ""}], tools)[: -len("<|user|>")]
        assert text.startswith(s), f"render of the first {i} messages is not a prefix"
        starts.append(len(s))
    starts.append(len(text))
    return starts


def char_roles(msgs, text, starts):
    """per-char (msg_index, role) arrays: header -1/system; assistant split into reasoning / tool_call / assistant."""
    n = len(text)
    mi = np.full(n, -1, dtype=np.int32)
    role = np.zeros(n, dtype=np.int8)  # 0 system 1 user 2 assistant 3 reasoning 4 tool_call 5 tool
    code = {"system": 0, "user": 1, "assistant": 2, "tool": 5}
    for i, m in enumerate(msgs):
        a, b = starts[i], starts[i + 1]
        mi[a:b] = i
        role[a:b] = code.get(m["role"], 0)
        if m["role"] == "assistant":
            seg = text[a:b]
            t0, t1 = seg.find("<think>"), seg.find("</think>")
            if t0 >= 0 and t1 >= 0:
                role[a + t0 : a + t1 + len("</think>")] = 3
            k = 0
            while (c0 := seg.find("<tool_call>", k)) >= 0:
                c1 = seg.find("</tool_call>", c0)
                c1 = len(seg) if c1 < 0 else c1 + len("</tool_call>")
                role[a + c0 : a + c1] = 4
                k = c1
    return mi, role


ROLE_NAMES = ["system", "user", "assistant", "reasoning", "tool_call", "tool"]


def view_rows(tk, text, msgs, tools, msg_map, tail_char, piece_cache, raw=False, first_reply=0, read_regex=None):
    """tokenize one view and return (ids, per-position dicts) for every position to read (+ the first `first_reply` tokens
    of the final assistant reply, whatever they are, + every token starting inside a `read_regex` match)."""
    enc = tk(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    n_full = len(ids)
    starts = spans(msgs, tools, text, raw)
    mi, role = char_roles(msgs, text, starts)
    reply_from = None
    if first_reply:
        a0 = starts[len(msgs) - 1]
        t1 = text.find("</think>", a0)
        reply_from = t1 + len("</think>") if t1 >= 0 else a0
    reply_seen = 0
    in_span = np.zeros(len(text) + 1, dtype=bool)
    if read_regex:
        for m in re.finditer(read_regex, text):
            in_span[m.start() : m.end()] = True
    marker_ids = {tk.convert_tokens_to_ids(m) for m in MARKERS}
    view_ids = ids[:CAP]
    turn = np.cumsum([t in marker_ids for t in view_ids]) - 1
    out = []
    for p in range(2, len(view_ids)):
        t = view_ids[p]
        if t not in piece_cache:
            piece_cache[t] = tk.decode([t])
        piece = piece_cache[t]
        special = t >= FIRST_ADDED
        early = reply_from is not None and offs[p][0] >= reply_from and reply_seen < first_reply
        if early:
            reply_seen += 1
        if not (special or is_punct(piece) or early or in_span[offs[p][0]]):
            continue
        c = offs[p][0]
        if c < tail_char:
            continue
        c = min(c, len(text) - 1)
        out.append(
            {
                "pos": p,
                "token": piece,
                "msg_index": int(msg_map(int(mi[c]))),
                "turn": int(turn[p]),
                "role": ROLE_NAMES[role[c]],
                "special": special,
            }
        )
    return ids, n_full, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--run-dir", required=True, help="runs/coding/<run> as published in the sprint dataset")
    ap.add_argument(
        "--realism", required=True, help="donehack/realism dir of a checkout with evilgenie/data_ib (rebuilds each arm's tool schemas)"
    )
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="ib_realistic_user")
    ap.add_argument("--part", default=None)
    ap.add_argument("--env", default=None, help="env column (default coding/<config>)")
    ap.add_argument("--positive", default="hack", help="row field used as `positive`")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rows-subset", default=None, help="JSON list of 0-based rows.jsonl lines to read (others skipped)")
    ap.add_argument("--raw-prefill", action="store_true", help="evalcue EVALCUE_RAW_PREFILL render (empty think block prefilled)")
    ap.add_argument("--first-reply-tokens", type=int, default=0, help="also read the first N tokens of the final reply")
    ap.add_argument("--read-regex", default=None, help="also read every token that starts inside a match of this regex")
    ap.add_argument("--tools-from", default="main-run variant schemas", help="labels_json provenance note for the tool schemas")
    a = ap.parse_args()
    R.TOK = None
    tk = R.tok(a.tok)
    rows = [json.loads(x) for x in Path(a.rows).read_text().splitlines()]
    if a.limit:
        rows = rows[: a.limit]
    tools_all = specs_for(rows, a.realism)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env_name = a.env or f"coding/{a.config}"
    piece_cache: dict[int, str] = {}
    seen: dict[bytes, tuple[str, int]] = {}
    eps, pos_rows, dups, views = [], [], [], []
    subset = set(json.loads(Path(a.rows_subset).read_text())) if a.rows_subset else None
    for ri, (r, tools) in enumerate(zip(rows, tools_all)):
        if subset is not None and ri not in subset:
            continue
        msgs = r["messages"]
        eid = f"{a.run_dir}#{ri}"
        tid0 = "cfi4:" + hashlib.md5(eid.encode()).hexdigest()[:12]
        text = raw_render(msgs) if a.raw_prefill else R.render(msgs, tools)
        cands = [("prefix", tid0, text, msgs, lambda i: i, 0)]
        n_tok = R.n_tokens(text)
        if n_tok > CAP:
            ltext, k = R.late_view(msgs, tools, CAP)
            if k:
                fu = next(i for i, m in enumerate(msgs) if m["role"] == "user")
                head = msgs[: fu + 1]
                lmsgs = (
                    head
                    + [{"role": "system", "content": f"[{len(msgs) - fu - 1 - k} earlier messages omitted for length]"}]
                    + msgs[len(msgs) - k :]
                )
                off = len(msgs) - k - (fu + 2)

                def mmap(i, fu=fu, off=off):
                    return i if i <= fu else (-1 if i == fu + 1 else i + off)

                tail_char = len(R.render(lmsgs[: fu + 2], tools))
                cands.append(("late", tid0 + ":late", ltext, lmsgs, mmap, tail_char))
        for view, tid, vtext, vmsgs, mmap, tail_char in cands:
            ids, n_full, prs = view_rows(
                tk, vtext, vmsgs, tools, mmap, tail_char, piece_cache, a.raw_prefill, a.first_reply_tokens, a.read_regex
            )
            truncated = n_full > CAP
            h = hashlib.blake2b(digest_size=16)
            arr = np.asarray(ids[:CAP], dtype=np.int32)
            keep, last = [], 0
            for d in prs:
                h.update(arr[last : d["pos"] + 1].tobytes())
                last = d["pos"] + 1
                key = h.copy().digest()
                if key in seen:
                    ct, cp = seen[key]
                    dups.append({"tid": tid, "pos": d["pos"], "canon_tid": ct, "canon_pos": cp, "canon_in": "readouts_dense"})
                    continue
                seen[key] = (tid, d["pos"])
                keep.append(d)
            rendered = vtext if not truncated else tk.decode(ids[:CAP])
            eps.append(
                {
                    "tid": tid,
                    "run_dir": a.run_dir,
                    "source_file": "rows.jsonl",
                    "row_index": ri,
                    "episode_id": eid,
                    "env": env_name,
                    "arm": r["arm"],
                    "item": r["item"],
                    "n_tokens_full": n_full,
                    "truncated": truncated,
                    "rendered_text": rendered,
                    "positions": np.asarray([d["pos"] for d in keep] + [x["pos"] for x in dups if x["tid"] == tid], dtype=np.int32),
                    "tools": "spec",
                }
            )
            final_in = not truncated
            views.append({"tid": tid, "ids": arr, "pos": np.asarray([d["pos"] for d in keep], dtype=np.int32)})
            for d in keep:
                p = d["pos"]
                pos_rows.append(
                    {
                        "config": a.config,
                        "run_dir": a.run_dir,
                        "source_file": "rows.jsonl",
                        "row_index": ri,
                        "episode_id": eid,
                        "env": env_name,
                        "arm": r["arm"],
                        "item": r["item"],
                        "sample": str(r.get("sample")),
                        "positive": str(r.get(a.positive)).lower() in ("true", "1"),
                        "positive_source": a.positive,
                        "aware": None,
                        "view": view,
                        "msg_index": d["msg_index"],
                        "turn": d["turn"],
                        "role": d["role"],
                        "kind": "dense",
                        "pos": p,
                        "token": d["token"],
                        "ctx_left40": tk.decode(ids[max(0, p - 40) : p + 1]),
                        "ctx_right40": tk.decode(ids[p + 1 : p + 41]),
                        "next16": tk.decode(ids[p + 1 : p + 17]),
                        "thinking": "off" if r.get("thinking") is False else "on",
                        "n_tokens_full": n_full,
                        "truncated": truncated,
                        "final_turn_in_view": final_in,
                        "labels_json": None,
                        "tid": tid,
                        "is_special": d["special"],
                    }
                )
        if ri % 50 == 0:
            print(f"{ri}/{len(rows)} rows, {len(pos_rows):,} positions, {len(dups):,} prefix dups", flush=True)
    extra = ({"part": a.part} if a.part else {}) | {"tools_from": a.tools_from}
    labels = {ri: labels_json(r, extra) for ri, r in enumerate(rows)}
    P = pd.DataFrame(pos_rows)
    P["labels_json"] = P.row_index.map(labels)
    if a.part:
        P["part"] = a.part
    P.to_parquet(out / "positions.parquet", index=False)
    pd.DataFrame(eps).to_parquet(out / "episodes.parquet", index=False)
    pd.DataFrame(dups, columns=["tid", "pos", "canon_tid", "canon_pos", "canon_in"]).to_parquet(
        out / "dense_prefix_dups.parquet", index=False
    )
    import pickle

    with open(out / "views.pkl", "wb") as f:
        pickle.dump(views, f)
    st = {
        "rows": len(rows),
        "views": len(views),
        "positions": len(P),
        "prefix_dups": len(dups),
        "special": int(P.is_special.sum()),
        "by_role": P.role.value_counts().to_dict(),
        "tokens_to_prefill": int(sum(len(v["ids"]) for v in views)),
    }
    (out / "prep_stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
