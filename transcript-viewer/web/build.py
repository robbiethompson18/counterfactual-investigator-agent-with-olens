"""The viewer's data: WeirdChat patterns and transcripts rendered through Qwen3.6-27B's chat template token by token,
the CHIVE incidents (web/chive.py), model organisms, GLM 5.3 rollouts, thinking resamples and confession follow-ups,
with every O-lens read we have (this machine's data/reads.jsonl and the other machines' shards pulled from HF).

scripts/publish.py turns this into the gzipped JSON the static page (web/index.html) fetches: `patterns`, `overview`,
`pattern_detail(i)` and `transcript(tid)` are the shapes of those files. There is no server: reads are batch jobs
(scripts/olens_read_requests.py at the repo root) whose outputs land in the HF dataset.
"""

import codecs
import glob
import json
import os
import re
import socket
from pathlib import Path

import chive
import extras
import followups
import glm
import organisms
import pyarrow.parquet as pq
import thinking
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HF = ROOT / "data/hf/data"
CACHE = ROOT / "data/reads.jsonl"
CLAUDE = ROOT / "data/claude.jsonl"  # scripts/claude_rollouts.py (deleted 2026-10-26, at 7296169): Haiku on the same prompts
STANCE = ROOT / "data/stance.jsonl"  # per-tid ENDORSE/CONDITIONAL/AGAINST from a Sonnet 5.5 stance classifier (3 votes)
# scripts/{followups,thinking}.py wrote these; both deleted 2026-10-26 (at commit 7296169)
FOLLOWUPS = ROOT / "data/followups.jsonl"  # "why did you do that?" turns on favourite transcripts
THINKING = ROOT / "data/thinking.jsonl"  # favourites resampled with thinking on
# The files above are this machine's; other machines' copies come from the HF dataset (scripts/publish.py) as
# data/shared/weirdchat/<kind>/<machine>.jsonl, and are read alongside them.
SHARED = ROOT / "data/shared/weirdchat"
MACHINE = os.environ.get("WCO_MACHINE") or socket.gethostname().split(".")[0].lower()
KINDS = {
    "reads": CACHE,
    "rollouts": CLAUDE,
    "stance": STANCE,
    "chive_judge": chive.JUDGE,
    "followups": FOLLOWUPS,
    "thinking": THINKING,
    "extras": extras.CACHE,
}
MODEL = "qwen/qwen3.6-27b"
LAYERS = list(range(20, 61, 4))
MAX_TOKENS = 4096
PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")  # stop tokens; copied in scripts/olens_read_requests.py (pod, own deps)
# GPT-2 byte-level BPE alphabet (bytes_to_unicode, inverted): token char -> raw byte
_KEEP = [*range(33, 127), *range(161, 173), *range(174, 256)]
BYTE = {chr(b): b for b in _KEEP} | {chr(256 + n): b for n, b in enumerate(b for b in range(256) if b not in _KEEP)}

S = {"reads": {}, "by_tid": {}}


# ---------------------------------------------------------------------- data
def load():
    pats = [p for p in pq.read_table(HF / "patterns.parquet").to_pylist() if p["subject_model"] == MODEL]
    prompts = {}
    for f in glob.glob(str(HF / "prompts/qwen__qwen3.6-27b/*.parquet")):
        for r in pq.read_table(f).to_pylist():
            prompts[r["prompt_id"]] = r
    trs = {}
    for f in glob.glob(str(HF / "transcripts/qwen__qwen3.6-27b/*.parquet")):
        for r in pq.read_table(f).to_pylist():
            trs[r["transcript_id"]] = r
    by_pat = {}
    for t in trs.values():
        by_pat.setdefault(t["pattern_id"], []).append(t["transcript_id"])
    for i, p in enumerate(pats):
        p["idx"] = i
        ids = by_pat.get(p["pattern_id"], [])
        # highlight first, then matched, then by prompt / sample
        ids.sort(key=lambda t: (not trs[t]["is_highlight"], not trs[t]["judgment"]["match"], t))
        p["tids"] = ids
        p["ds"] = "weirdchat"
    S.update(patterns=pats + chive.load(len(pats), shards("chive_judge")), prompts=prompts, trs=trs)


def render(tid):
    """Chat-template the transcript, tokenise with offsets, and locate messages, citations and
    the end-of-turn zones (<|im_end|> and the template tokens up to the next content)."""
    if tid in S.setdefault("rendered", {}):
        return S["rendered"][tid]
    if chive.has(tid):
        ids = chive.C["tokens"][tid]["token_ids"]
        S["rendered"][tid] = chive.render(tid, decode(ids), [x in S["special"] for x in ids])
        return S["rendered"][tid]
    tok, t = S["tok"], S["trs"][tid]
    if "ids" in t:  # model organisms (organisms.py): the token ids the lens read, message ranges precomputed
        ids, msg_tok, cites, truncated = t["ids"], t["msg_tok"], [], False
    else:
        ids, msg_tok, cites, truncated = render_text(tid, tok, t)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    zones = []  # [start, end) token ranges: an <|im_end|> through the next content
    starts = [r[0] for r in msg_tok if r] + [len(ids)]
    for i, x in enumerate(ids):
        if x == im_end:
            nxt = min([s for s in starts if s > i] or [len(ids)])
            zones.append([i, min(nxt, len(ids))])
    out = {
        "tid": tid,
        "ids": ids,
        "tokens": decode(ids),
        "special": [x in S["special"] for x in ids],
        "msgs": msg_tok,
        "roles": [m["role"] for m in t["messages"]],
        "cites": cites,
        "zones": zones,
        "truncated": truncated,
        # the end-of-turn zone the list previews and a transcript opens on: after the user's (last) message.
        # Organisms have a system prompt first and follow-ups a second user turn, so it's the second-to-last zone there;
        # the same for thinking resamples, unless the reply hit the cap and has no <|im_end|> of its own
        "main_zone": len(zones) - (1 if t.get("thinking", {}).get("hit_cap") else 2)
        if {"organism", "followup", "thinking"} & t.keys()
        else 0,
        "focus": [0, len(t["messages"]) - 1],  # messages whose punctuation is worth reading
    }
    S["rendered"][tid] = out
    return out


def render_text(tid, tok, t):
    """WeirdChat transcripts: ids, message token ranges and judge citations, from the chat-templated text."""
    msgs = [{"role": m["role"], "content": m["content"]} for m in t["messages"]]
    text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc.input_ids[:MAX_TOKENS], enc.offset_mapping[:MAX_TOKENS]
    # message content char ranges in the rendered text
    spans, at = [], 0
    for m in msgs:
        c = m["content"].strip()  # the template strips surrounding whitespace
        s = text.find(c, at)
        assert s >= 0, f"message content not found in the rendered transcript {tid}"
        spans.append((s, s + len(c)))
        at = s + len(c)
        m["lead"] = m["content"].find(c)  # citation offsets index the unstripped content

    def tok_range(a, b):
        idx = [i for i, (x, y) in enumerate(offs) if y > a and x < b]
        return [idx[0], idx[-1] + 1] if idx else None

    msg_tok = [tok_range(a, b) for a, b in spans]
    cites = []
    for c in t.get("citations") or []:
        a0 = spans[-1][0] - msgs[-1]["lead"]  # citations index the assistant reply
        r = tok_range(a0 + c["start_idx"], a0 + c["end_idx"])
        if r:
            cites.append({"range": r, "note": c["note"], "quote": c["quote"]})
    return ids, msg_tok, cites, len(enc.input_ids) > MAX_TOKENS


def decode(ids):
    """Token strings. Byte-level BPE splits multi-byte characters (emoji) across tokens, and decoding one id alone turns
    each piece into U+FFFD: decode the raw bytes incrementally, so a character shows on the token holding its last byte."""
    utf8 = codecs.getincrementaldecoder("utf-8")("replace")
    return [utf8.decode(bytes(BYTE[c] for c in piece)) for piece in S["tok"].convert_ids_to_tokens(ids)]


# ---------------------------------------------------------------------- reads
LENS = "olens_s3d_space_rl600"  # the current lens; every read made since 2026-10-23 is tagged and keyed with it
OLD_LENS = "olens_s3d_rl600"  # the Chinese-prefix-bug lens: every cached read without a "lens" field (all before 2026-10-23)


def key(tid, pos, layer, temperature=0.0, n=1, seed=0, lens=LENS):
    """Old-lens reads keep their lens-less keys, so a key from here never finds them."""
    return f"{tid}|{pos}|{layer}|{float(temperature):g}|{n}|{seed}" + ("" if lens == OLD_LENS else f"|{lens}")


def ingest(r):
    r.setdefault("lens", OLD_LENS)
    S["reads"][r["key"]] = r
    S["by_tid"].setdefault(r["tid"], {})[r["key"]] = r


def tid_reads(tid):
    """One read per (pos, layer, sampling): the current lens's, else the old lens's (the page marks those)."""
    best = {}
    for r in S["by_tid"].get(tid, {}).values():
        k = (r["pos"], r["layer"], r["temperature"], r["n"], r["seed"])
        if k not in best or r["lens"] == LENS:
            best[k] = r
    return list(best.values())


def overview():
    """Per pattern: the greedy L40 reads in the highlight transcript's main end-of-turn
    zone (after the user's last message), for the list."""
    out = {}
    for p in S["patterns"]:
        if not p["tids"]:
            continue
        tid = p["tids"][0]
        r = render(tid)
        z = r["zones"][r["main_zone"] :][:1]
        rows = [
            r for r in tid_reads(tid) if r["temperature"] == 0 and r["n"] == 1 and r["layer"] == 40 and z and z[0][0] <= r["pos"] < z[0][1]
        ]
        if rows:
            out[p["idx"]] = [
                {
                    "pos": r["pos"],
                    "layer": r["layer"],
                    "token": r["token"],
                    "bullets": r["samples"][0]["bullets"][:4],
                    "coefs": r["samples"][0].get("coefs", [])[:4],  # organism reads have no FVE yet
                    "bfve": r["samples"][0].get("bfve", [])[:4],
                    "fve": r["samples"][0].get("fve"),
                }
                for r in sorted(rows, key=lambda r: (r["pos"], r["layer"]))
            ]
    return out


def punct_positions(r, msg=-1):
    a, b = r["msgs"][msg]
    return [i for i in range(a, b) if PUNCT.match(r["tokens"][i])]


# ---------------------------------------------------------------------- patterns and transcripts
def pattern_summary(p):
    if p["ds"] == "chive":
        return chive.summary(p)
    e = p["elo"] or {}
    return {
        "idx": p["idx"],
        "ds": p["ds"],
        "title": p["title"],
        "behavior": p["behavior_id"],
        "method": p["method"],
        "match_rate": (p["metrics"] or {}).get("match_rate"),
        "n_prompts": p["n_prompts"],
        "n_transcripts": len(p["tids"]),
        "url": p["weirdchat_url"],
        "elo": {k: round(v["elo"]) for k, v in e.items() if isinstance(v, dict) and v.get("elo") is not None},
        "elo_mean": e.get("mean"),
        "search": p["description"] + " " + p["representative_user_text"],
        **{k: p[k] for k in ("organism", "followup", "thinking", "glm") if k in p},
    }


def pattern_detail(i):
    p = S["patterns"][i]
    if p["ds"] == "chive":
        return chive.detail(p)
    trs = []
    pidx = {}
    for tid in p["tids"]:
        t = S["trs"][tid]
        pidx.setdefault(t["prompt_id"], len(pidx))
        trs.append(
            {
                "tid": tid,
                "prompt": pidx[t["prompt_id"]],
                "match": t["judgment"]["match"],
                "highlight": t["is_highlight"],
                "explanation": t["judgment"]["explanation"],
                "preview": t["messages"][-1]["content"][:160] if t["messages"] else "",
                **({"organism": {k: t["organism"][k] for k in ("key", "kind", "prompt", "sample", "read")}} if "organism" in t else {}),
                **({"followup": t["followup"]} if "followup" in t else {}),
                **({"thinking": t["thinking"]} if "thinking" in t else {}),
                **({"glm": t["glm"]} if "glm" in t else {}),
            }
        )
    # behaviour rate per prompt: judged rollouts of that prompt (~64 each) that matched
    prompts = []
    for pid, k in pidx.items():
        ts = [t for t in trs if t["prompt"] == k]
        pr = S["prompts"].get(pid) or {}
        user = (
            next((m["content"] for m in (pr.get("messages") or []) if m["role"] == "user"), None)
            or (S["trs"][ts[0]["tid"]]["messages"] or [{"content": pid.split(":")[-1]}])[0]["content"]
        )
        prompts.append(
            {
                "idx": k,
                "n": len(ts),
                "matched": sum(t["match"] for t in ts),
                "text": user[:400],
                "highlight": any(t["highlight"] for t in ts),
            }
        )
    return dict(
        pattern_summary(p),
        description=p["description"],
        cluster=p["cluster_label"],
        representative=p["representative_user_text"],
        metrics=p["metrics"],
        openrouter=p["openrouter_replication"],
        transcripts=trs,
        prompts=prompts,
    )


def shards(kind):
    """Other machines' files for a kind, then this machine's (last, so its records win)."""
    other = [p for p in sorted((SHARED / kind).glob("*.jsonl")) if p.stem != MACHINE]
    return [p for p in [*other, KINDS[kind]] if p.exists()]


def merged(kind, index):
    """index(records of a kind across all shards), computed once per kind."""
    if kind not in S.setdefault("merged", {}):
        S["merged"][kind] = index([json.loads(line) for p in shards(kind) for line in p.open()])
    return S["merged"][kind]


def claude_rollouts(prompt_id):
    """Claude rollouts on a transcript's prompt (scripts/claude_rollouts.py)."""

    def by_prompt(rs):
        out = {}
        for r in rs:
            out.setdefault(r["prompt_id"], []).append(r)
        return out

    return merged("rollouts", by_prompt).get(prompt_id, [])


def stance(tid):
    """Stance label for a transcript."""
    return merged("stance", lambda rs: {r["tid"]: r["stance"] for r in rs}).get(tid)


def transcript(tid):
    r = render(tid)
    out = {k: v for k, v in r.items() if k != "ids"}
    out.update(punct=[i for k in r["focus"] for i in punct_positions(r, k)], reads=tid_reads(tid), lens=LENS)
    if chive.has(tid):
        return dict(out, **chive.transcript_extra(tid))
    t = S["trs"][tid]
    out.update(
        ds="weirdchat",
        match=t["judgment"]["match"],
        explanation=t["judgment"]["explanation"],
        highlight=t["is_highlight"],
        url=t["weirdchat_url"],
        claude=claude_rollouts(t["prompt_id"]),
        stance=stance(tid),
        extras=sorted(extras.of(tid)),
        prompt_rate=prompt_rate(t["prompt_id"]),
    )
    if "followup" in t:
        out["followup"] = t["followup"]
    if "thinking" in t:
        out["thinking"] = t["thinking"]
    if "organism" in t:  # reads on base Qwen's activations over the same tokens; the response tokens that were read
        zone = {i for a, b in r["zones"] for i in range(a, b)}
        out.update(
            organism=t["organism"],
            base_reads=tid_reads(tid + organisms.BASE),
            punct=[p for p in t["organism"]["read_pos"] if p not in zone],
            user_punct=0,
        )
    if "glm" in t:  # Celeste's GLM-5.3 lens, one layer, read offline at her chosen positions (web/glm.py)
        out.update(ds="glm", glm=t["glm"], punct=t["glm"]["read_pos"], user_punct=0, lens=glm.LENS, layers=[glm.LAYER])
    return out


def prompt_rate(pid):
    pr = S["prompts"].get(pid) or {}
    m = pr.get("match_summary") or {}
    return {"n": m.get("n_samples"), "matched": m.get("n_matched")} if m else None


def init():
    """Tokenizer, WeirdChat, the extra datasets and every read (the shared shards, then this machine's, which wins)."""
    S["tok"] = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    S["special"] = set(S["tok"].all_special_ids) | {
        S["tok"].convert_tokens_to_ids(x) for x in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
    }
    load()
    organisms.load(S, key)
    glm.load(S, key)
    followups.load(S, [json.loads(line) for p in shards("followups") for line in p.open()])
    thinking.load(S, [json.loads(line) for p in shards("thinking") for line in p.open()])
    for p in shards("reads"):
        for line in p.open():
            try:
                ingest(json.loads(line))
            except ValueError:  # a torn last line from an interrupted writer
                pass
    extras.load(shards("extras"))
