"""O-lens viewer: http://<host>:8803/ (binds to $WCO_BIND, default 127.0.0.1)

Browses the Qwen3.6-27B patterns of Transluce/WeirdChat with their transcripts rendered through the
model's chat template, token by token, and the CHIVE incidents (web/chive.py) with Agam's per-token
readout banks. Clicking a token reads it with the agu18dec oracle lens
on the H100 pod (web/pod.py); every read is cached in data/reads.jsonl, so anything read once
stays browsable when the pod is down.

usage: .venv/bin/python web/server.py [--port 8803] [--no-pod]
"""

import argparse
import codecs
import glob
import json
import os
import re
import socket
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import chive
import organisms
import extras
import steer
from judge import Judge
import pyarrow.parquet as pq
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HF = ROOT / "data/hf/data"
CACHE = ROOT / "data/reads.jsonl"
CLAUDE = ROOT / "data/claude.jsonl"  # scripts/claude_rollouts.py: Haiku on the same prompts
STANCE = ROOT / "data/stance.jsonl"  # per-tid ENDORSE/CONDITIONAL/AGAINST from a Sonnet 5.5 stance classifier (3 votes)
# The files above are this machine's; other machines' copies come from the HF dataset (scripts/publish.py) as
# data/shared/weirdchat/<kind>/<machine>.jsonl, and are read alongside them.
SHARED = ROOT / "data/shared/weirdchat"
MACHINE = os.environ.get("WCO_MACHINE") or socket.gethostname().split(".")[0].lower()
KINDS = {"reads": CACHE, "rollouts": CLAUDE, "stance": STANCE, "chive_judge": chive.JUDGE}
MODEL = "qwen/qwen3.6-27b"
LAYERS = list(range(20, 61, 4))
MAX_TOKENS = 4096
PRE_LAYERS = [40]  # precomputed on every highlight transcript (see precompute())
PRE_BATCH = 96
NEWLINE = re.compile(r"^\s*\n\s*$")
PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")
# GPT-2 byte-level BPE alphabet (bytes_to_unicode, inverted): token char -> raw byte
_KEEP = [*range(33, 127), *range(161, 173), *range(174, 256)]
BYTE = {chr(b): b for b in _KEEP} | {chr(256 + n): b for n, b in enumerate(b for b in range(256) if b not in _KEEP)}

S = {"reads": {}, "by_tid": {}, "pre": {"state": "idle", "done": 0, "total": 0}}
LOCK = threading.Lock()
POD = None
JPP_LOCAL = None  # web/run_lens.sh --extras jpp on this machine, for J++ only


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
        # Organisms have a system prompt first, so it's the second-to-last zone there
        "main_zone": len(zones) - 2 if "organism" in t else 0,
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


# ---------------------------------------------------------------------- reads + cache
def key(tid, pos, layer, temperature=0.0, n=1, seed=0):
    return f"{tid}|{pos}|{layer}|{float(temperature):g}|{n}|{seed}"


def load_cache():
    """Read whatever was appended to data/reads.jsonl since the last call (scripts/fleet.py writes
    to it from outside); a trailing partial line waits for the next call."""
    if not CACHE.exists():
        return
    with LOCK:
        with open(CACHE, "rb") as f:
            f.seek(S.get("cache_off", 0))
            buf = f.read()
        end = buf.rfind(b"\n") + 1
        S["cache_off"] = S.get("cache_off", 0) + end
        new = []
        for line in buf[:end].splitlines():
            try:
                r = json.loads(line)
                S["reads"][r["key"]] = r
                S["by_tid"].setdefault(r["tid"], {})[r["key"]] = r
                new.append(r["key"])
            except ValueError:
                pass
    if "queue" in S and new:
        S["queue"].finished(new)


def follow_cache():
    while True:
        time.sleep(15)
        load_cache()


def save(recs):
    with LOCK:
        with open(CACHE, "a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in recs))
        for r in recs:
            S["reads"][r["key"]] = r
            S["by_tid"].setdefault(r["tid"], {})[r["key"]] = r
    if "queue" in S:
        S["queue"].finished([r["key"] for r in recs])


def read(tid, items, temperature=0.0, n=1, seed=0, cached_only=False):
    """items [{pos, layer}] -> records, from the cache or the pod (one capture per call)."""
    r = render(tid)
    keys = [key(tid, int(it["pos"]), int(it["layer"]), temperature, n, seed) for it in items]
    todo = [(k, it) for k, it in zip(keys, items) if k not in S["reads"]]
    todo = list({k: it for k, it in todo}.items())
    if todo and not cached_only:
        no_live_reads(tid)
        if not (POD and POD.ready()):
            raise RuntimeError("the H100 isn't running: start it from the header (cached reads still show)")
        res = POD.call("/read", {"ids": r["ids"], "items": [it for _, it in todo], "temperature": temperature, "n": n, "seed": seed})
        save(
            [
                {
                    "key": k,
                    "tid": tid,
                    "pos": int(it["pos"]),
                    "layer": int(it["layer"]),
                    "temperature": temperature,
                    "n": n,
                    "seed": seed,
                    "token": r["tokens"][int(it["pos"])],
                    "norm": res["norms"][j],
                    "samples": res["results"][j],
                    "t": time.time(),
                }
                for j, (k, it) in enumerate(todo)
            ]
        )
    return [S["reads"][k] for k in keys if k in S["reads"]]


def tid_reads(tid):
    return list(S["by_tid"].get(tid, {}).values())


def no_live_reads(tid):
    if "organism" in S["trs"][tid]:
        raise RuntimeError("model-organism transcripts are read offline, with the organism's LoRA (scripts/organism_olens.py)")


def overview():
    """Per pattern: the cached greedy L40 reads in the highlight transcript's main end-of-turn
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


def precompute():
    """Highlight transcripts at PRE_LAYERS: first every end-of-turn zone token, then every
    punctuation token in the assistant turn. Cached reads are skipped; the rest go to the pod in
    batches of ~PRE_BATCH reads across transcripts (one capture per transcript). Every batch is
    appended to data/reads.jsonl as it lands. When everything is done, the pod is torn down."""
    S["pre"].update(state="running", done=0, error=None)
    todo = []  # (phase, tid, item)
    for phase in ("zones", "punct"):
        for p in S["patterns"]:
            if not p["tids"] or "organism" in p:
                continue
            tid = p["tids"][0]
            r = render(tid)
            pos = (
                [i for a, b in r["zones"] for i in range(a, b)]
                if phase == "zones"
                else [i for k in r["focus"] for i in punct_positions(r, k)]
            )
            todo += [(phase, tid, {"pos": i, "layer": L}) for i in pos for L in PRE_LAYERS]
    S["pre"]["total"] = len(todo)
    todo_left = [t for t in todo if key(t[1], t[2]["pos"], t[2]["layer"]) not in S["reads"]]
    S["pre"]["done"] = len(todo) - len(todo_left)
    try:
        for k in range(0, len(todo_left), PRE_BATCH):
            chunk = todo_left[k : k + PRE_BATCH]
            S["pre"]["phase"] = chunk[0][0]
            if not POD.ready():
                raise RuntimeError("lens backend went away")
            jobs = {}
            for _, tid, it in chunk:
                jobs.setdefault(tid, []).append(it)
            res = POD.call("/read_multi", {"jobs": [{"ids": render(t)["ids"], "items": its} for t, its in jobs.items()]})
            recs = []
            for (tid, its), out, norms in zip(jobs.items(), res["results"], res["norms"]):
                toks = render(tid)["tokens"]
                recs += [
                    {
                        "key": key(tid, it["pos"], it["layer"]),
                        "tid": tid,
                        "pos": it["pos"],
                        "layer": it["layer"],
                        "temperature": 0.0,
                        "n": 1,
                        "seed": 0,
                        "token": toks[it["pos"]],
                        "norm": norms[j],
                        "samples": out[j],
                        "t": time.time(),
                    }
                    for j, it in enumerate(its)
                ]
            save(recs)
            S["pre"]["done"] += len(chunk)
        rescore()
        S["pre"]["state"] = "done"
    except Exception as e:  # noqa: BLE001  (server/worker boundary: report and keep going)
        S["pre"].update(state="error", error=f"{type(e).__name__}: {e}")


def rescore():
    """Cached reads from before per-bullet FVE existed: re-score their texts on the pod (one
    capture per transcript, no decoding) and append the updated records."""
    old = [r for r in S["reads"].values() if r["samples"] and "bfve" not in r["samples"][0]]
    S["pre"].update(phase="rescore", done=0, total=len(old))
    by_tid = {}
    for r in old:
        by_tid.setdefault(r["tid"], []).append(r)
    batch = []

    def flush():
        res = POD.call(
            "/rescore",
            {
                "jobs": [
                    {
                        "ids": render(t)["ids"],
                        "items": [{"pos": r["pos"], "layer": r["layer"], "texts": [x["text"] for x in r["samples"]]} for r in rs],
                    }
                    for t, rs in batch
                ]
            },
        )
        recs = []
        for (_, rs), out in zip(batch, res["results"]):
            for r, samples in zip(rs, out):
                recs.append(dict(r, samples=samples))
        save(recs)
        S["pre"]["done"] += len(recs)
        batch.clear()

    for tid, rs in by_tid.items():
        for k in range(0, len(rs), 256):
            batch.append((tid, rs[k : k + 256]))
            if sum(len(x) for _, x in batch) >= 256:
                flush()
    if batch:
        flush()


def autoprecompute():
    """Start (and after a backend restart, resume) the precompute whenever the backend is ready."""
    while True:
        time.sleep(30)
        if S["pre"]["state"] in ("idle", "error") and POD and POD.ready():
            precompute()


# ---------------------------------------------------------------------- http
# ---------------------------------------------------------------------- queue
QUEUE_LAYERS = [24, 32, 40, 48, 56]


def relevant_positions(tid):
    """End-of-turn zones + punctuation and newlines in the focus messages (WeirdChat: both turns): where the lens reads best."""
    r = render(tid)
    newlines = {i for k in r["focus"] if r["msgs"][k] for i in range(*r["msgs"][k]) if NEWLINE.match(r["tokens"][i])}
    return sorted({i for a, b in r["zones"] for i in range(a, b)} | {i for k in r["focus"] for i in punct_positions(r, k)} | newlines)


def queue_items(tid, positions, layers, src):
    toks = render(tid)["tokens"]
    return [{"key": key(tid, p, L), "tid": tid, "pos": p, "layer": L, "src": src} for p in positions for L in layers if 0 <= p < len(toks)]


def title_of(tid):
    if chive.has(tid):
        return f"CHIVE {tid}"
    t = S["trs"].get(tid)
    if not t:
        return tid
    p = S["pat_by_id"].get(t["pattern_id"])
    return f"{p['title'] if p else t['pattern_id']} · {tid.split('/')[-1]}"


def read_chunk(items):
    """Read queue items on the viewer's own pod: one capture per transcript."""
    jobs = {}
    for it in items:
        jobs.setdefault(it["tid"], []).append({"pos": it["pos"], "layer": it["layer"]})
    res = POD.call("/read_multi", {"jobs": [{"ids": render(t)["ids"], "items": its} for t, its in jobs.items()]})
    recs = []
    for (tid, its), out, norms in zip(jobs.items(), res["results"], res["norms"]):
        toks = render(tid)["tokens"]
        recs += [
            {
                "key": key(tid, it["pos"], it["layer"]),
                "tid": tid,
                "pos": it["pos"],
                "layer": it["layer"],
                "temperature": 0.0,
                "n": 1,
                "seed": 0,
                "token": toks[it["pos"]],
                "norm": norms[j],
                "samples": out[j],
                "t": time.time(),
            }
            for j, it in enumerate(its)
        ]
    save(recs)


def consume():
    """The viewer's own pod (Start GPU) works the queue too, between interactive reads."""
    while True:
        time.sleep(3)
        if not (POD and POD.ready()) or S["pre"]["state"] == "running":
            continue
        items = S["queue"].take(96, "viewer-pod")
        if not items:
            continue
        try:
            read_chunk(items)
        except Exception as e:  # noqa: BLE001  (server/worker boundary: report and keep going)
            print(f"queue chunk failed on the viewer pod: {e}", flush=True)
            S["queue"].release([it["key"] for it in items])
            time.sleep(30)


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
        **({"organism": p["organism"]} if "organism" in p else {}),
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
                "preview": t["messages"][-1]["content"][:160],
                **({"organism": {k: t["organism"][k] for k in ("key", "kind", "prompt", "sample", "read")}} if "organism" in t else {}),
            }
        )
    # behaviour rate per prompt: judged rollouts of that prompt (~64 each) that matched
    prompts = []
    for pid, k in pidx.items():
        ts = [t for t in trs if t["prompt"] == k]
        pr = S["prompts"].get(pid) or {}
        user = (
            next((m["content"] for m in (pr.get("messages") or []) if m["role"] == "user"), None)
            or S["trs"][ts[0]["tid"]]["messages"][0]["content"]
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
    """index(records of a kind across all shards), recomputed when any shard changes."""
    paths = shards(kind)
    sig = [(p, p.stat().st_mtime) for p in paths]
    if S.setdefault("merged", {}).get(kind, (None,))[0] != sig:
        S["merged"][kind] = (sig, index([json.loads(line) for p in paths for line in p.open()]))
    return S["merged"][kind][1]


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
    out.update(
        queued=[[int(k.split("|")[1]), int(k.split("|")[2])] for k in S["queue"].pending_keys(tid)] if "queue" in S else [],
        relevant=len(relevant_positions(tid)),
        punct=[i for k in r["focus"] for i in punct_positions(r, k)],
        reads=tid_reads(tid),
    )
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
    if "organism" in t:  # reads on base Qwen's activations over the same tokens; the response tokens that were read
        zone = {i for a, b in r["zones"] for i in range(a, b)}
        out.update(
            organism=t["organism"],
            base_reads=tid_reads(tid + organisms.BASE),
            punct=[p for p in t["organism"]["read_pos"] if p not in zone],
            user_punct=0,
        )
    return out


def prompt_rate(pid):
    pr = S["prompts"].get(pid) or {}
    m = pr.get("match_summary") or {}
    return {"n": m.get("n_samples"), "matched": m.get("n_matched")} if m else None


def prompt_ids(tid):
    """The rendered transcript up to the start of the assistant's reply (generation prompt included)."""
    r = render(tid)
    return r["ids"][: r["msgs"][-1][0]]


def user_messages(tid):
    return [{"role": m["role"], "content": m["content"]} for m in S["trs"][tid]["messages"][:-1]]


def judge_for(tid):
    return Judge.from_env(S["rubrics"][S["trs"][tid]["behavior_id"]])


def judge_calib(tid):
    """Judge WeirdChat's own rollouts of this transcript's prompt and compare with its labels."""
    t = S["trs"][tid]
    S.setdefault("calib", {})[tid] = {"state": "running"}
    try:
        j = judge_for(tid)
        if j is None:
            raise RuntimeError("no judge configured (WCO_JUDGE_URL)")
        sib = [
            x for x in S["trs"].values() if x["prompt_id"] == t["prompt_id"] and x["judgment"]["match"] is not None
        ]
        out = j.judge_many(user_messages(tid), [x["messages"][-1]["content"] or "" for x in sib])
        pairs = [(bool(x["judgment"]["match"]), o["match"]) for x, o in zip(sib, out) if o["match"] is not None]
        n = len(pairs)
        agree = sum(a == b for a, b in pairs) / n if n else None
        pa, pb = (sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n) if n else (0, 0)
        pe = pa * pb + (1 - pa) * (1 - pb)
        S["calib"][tid] = {"state": "done", "n": n, "weirdchat_rate": pa, "judge_rate": pb, "agree": agree,
                           "kappa": (agree - pe) / (1 - pe) if n and pe < 1 else None}
    except Exception as e:  # noqa: BLE001  (worker boundary: report and keep going)
        S["calib"][tid] = {"state": "error", "error": f"{type(e).__name__}: {e}"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200):
        b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self.send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/patterns":
                return self.send({"patterns": [pattern_summary(p) for p in S["patterns"]], "layers": LAYERS, "pre_layers": PRE_LAYERS})
            if u.path == "/api/overview":
                return self.send(overview())
            if u.path == "/api/pattern":
                return self.send(pattern_detail(int(q["i"])))
            if u.path == "/api/transcript":
                return self.send(transcript(q["tid"]))
            if u.path == "/api/locate":  # transcript -> its pattern index
                pid = q["tid"] if chive.has(q["tid"]) else S["trs"][q["tid"]]["pattern_id"]
                return self.send({"pattern": S["pat_by_id"][pid]["idx"]})
            if u.path == "/api/bank":  # one reader's cells for a chunk of positions, all layers
                return self.send(chive.bank_chunk(q["tid"], q["reader"], int(q["chunk"])))
            if u.path == "/api/queue":
                return self.send(S["queue"].summary(title_of))
            if u.path == "/style.css":  # the viewer's stylesheet, shared with the case page
                h = (HERE / "index.html").read_text()
                return self.send(h[h.index("<style>") + 7 : h.index("</style>")], "text/css; charset=utf-8")
            if u.path in ("/case", "/case.html"):  # one transcript, every lens side by side
                return self.send((HERE / "case.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/vectors":
                return self.send({"vectors": steer.vectors(), "judge": bool(os.environ.get("WCO_JUDGE_URL"))})
            if u.path == "/api/steer":  # runs for a transcript's prompt; texts only with ?full=1
                rs = steer.runs(q["tid"])
                if not q.get("full"):
                    rs = [{k: v for k, v in r.items() if k not in ("texts", "explanations")} for r in rs]
                return self.send({"runs": rs, "calib": S.get("calib", {}).get(q["tid"])})
            if u.path == "/api/extras_all":  # every cached NLA / J++ read of a transcript
                return self.send(extras.of(q["tid"]))
            if u.path == "/api/extras":  # cached NLA / J++ reads of one token
                return self.send(extras.of(q["tid"]).get(int(q["pos"]), {}))
            if u.path == "/api/status":
                return self.send(
                    {
                        "pod": POD.status() if POD else {"state": "disabled"},
                        "pre": S["pre"],
                        "n_reads": len(S["reads"]),
                        "jpp_local": bool(JPP_LOCAL and JPP_LOCAL.ready()),
                        "queue": {k: v for k, v in S["queue"].summary(title_of, top=0).items() if k != "rows"},
                    }
                )
            self.send({"error": "not found"}, code=404)
        except Exception as e:  # noqa: BLE001  (server/worker boundary: report and keep going)
            traceback.print_exc()
            self.send({"error": f"{type(e).__name__}: {e}"}, code=500)

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            p = self.path
            if p == "/api/read":
                recs = read(req["tid"], req["items"], float(req.get("temperature", 0)), int(req.get("n", 1)), int(req.get("seed", 0)))
                return self.send({"reads": recs})
            if p == "/api/score_edit":  # FVE of edited lens text: {tid, pos, layer, text, lens: olens|nla}
                if not (POD and POD.ready()):
                    raise RuntimeError("the GPU pod isn't running: start it from the header")
                ids = render(req["tid"])["ids"]
                if req.get("lens") == "nla":
                    return self.send(POD.call("/nla_score", {"ids": ids, "pos": int(req["pos"]), "text": req["text"]}))
                res = POD.call("/rescore", {"jobs": [{"ids": ids, "items": [{"pos": int(req["pos"]), "layer": int(req["layer"]), "texts": [req["text"]]}]}]})
                return self.send(res["results"][0][0][0])
            if p == "/api/contrib":  # per-block attention / MLP writes at one token: {tid, pos}
                if not (POD and POD.ready()):
                    raise RuntimeError("the GPU pod isn't running: start it from the header")
                vecs = {f"{v['name']} (L{v['layer']})": steer.V[v["vid"]]["v"] for v in steer.vectors()}
                return self.send(POD.call("/contrib", {"ids": render(req["tid"])["ids"], "pos": int(req["pos"]), "vecs": vecs, "layered": req.get("layered") or {}}))
            if p == "/api/vector":
                return self.send(steer.build(POD, req.get("name"), req["spec"]))
            if p == "/api/vector/delete":
                steer.delete(req["vid"])
                return self.send({"ok": True})
            if p == "/api/probe":
                return self.send(steer.probe(POD, req["tid"], render(req["tid"])["ids"], req["vid"]))
            if p == "/api/steer":
                tid = req["tid"]
                return self.send(steer.start(POD, judge_for(tid), tid, prompt_ids(tid), user_messages(tid), req.get("vid"), req.get("alpha", 0), int(req.get("n", 64)), ablate=req.get("ablate") or ()))
            if p == "/api/steer/rejudge":  # re-run failed judge calls of a finished run
                r = steer.R[req["rid"]]
                return self.send(steer.rejudge(judge_for(r["tid"]), req["rid"], user_messages(r["tid"])))
            if p == "/api/judge_calib":  # judge WeirdChat's stored rollouts of this prompt, compare labels
                tid = req["tid"]
                threading.Thread(target=judge_calib, args=(tid,), daemon=True).start()
                return self.send({"started": True})
            if p == "/api/extras":  # NLA + J++ on the pod: {tid, positions, kinds?}
                r = render(req["tid"])
                pos = [int(x) for x in req["positions"]]
                kinds = tuple(req.get("kinds") or ("nla", "jpp"))
                out = {}
                for k in kinds:  # J++ can run on this machine's GPU (--jpp-local), the NLA needs the pod
                    pod = JPP_LOCAL if k == "jpp" and JPP_LOCAL and JPP_LOCAL.ready() else POD
                    for q, x in extras.read(pod, S["tok"], req["tid"], r["ids"], pos, (k,)).items():
                        out.setdefault(q, {}).update(x)
                return self.send({"extras": out})
            if p == "/api/pod/start" and POD:
                return self.send(dict(started=POD.start(), **POD.status()))
            if p == "/api/pod/stop" and POD:
                return self.send(dict(stopped=POD.stop(), **POD.status()))
            if p == "/api/queue/add":  # from the UI: to the front
                tid = req["tid"]
                no_live_reads(tid)
                layers = [int(L) for L in req.get("layers") or QUEUE_LAYERS if int(L) in LAYERS]
                scope = req.get("scope", "relevant")
                if scope == "relevant":
                    pos = relevant_positions(tid)
                elif scope == "all":  # every token of the rendered transcript
                    pos = list(range(len(render(tid)["ids"])))
                else:
                    pos = [int(x) for x in req["positions"]]
                n = S["queue"].add(queue_items(tid, pos, layers, "user"), front=True)
                return self.send(dict(added=n, **S["queue"].summary(title_of)))
            if p == "/api/queue/bulk":  # from scripts/fleet.py: to the back
                its = [
                    {
                        "key": key(x["tid"], x["pos"], x["layer"]),
                        "tid": x["tid"],
                        "pos": int(x["pos"]),
                        "layer": int(x["layer"]),
                        "src": "bulk",
                    }
                    for x in req["items"]
                ]
                return self.send({"added": S["queue"].add(its)})
            if p == "/api/queue/take":
                mx = int(req.get("max_tokens") or 0)
                fits = (lambda it: len(render(it["tid"])["ids"]) <= mx) if mx else None
                return self.send({"items": S["queue"].take(int(req.get("n", 96)), req.get("worker", "?"), fits)})
            if p == "/api/queue/release":
                S["queue"].release(req["keys"])
                return self.send({"ok": True})
            if p == "/api/precompute":
                if S["pre"]["state"] != "running":
                    threading.Thread(target=precompute, daemon=True).start()
                return self.send(S["pre"])
            self.send({"error": "not found"}, code=404)
        except Exception as e:  # noqa: BLE001  (server/worker boundary: report and keep going)
            traceback.print_exc()
            self.send({"error": f"{type(e).__name__}: {e}"}, code=500)


class Server6(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def init():
    """Tokenizer, WeirdChat and every cached read (this machine's, then the shared shards): all but HTTP and GPUs."""
    S["tok"] = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    S["special"] = set(S["tok"].all_special_ids) | {
        S["tok"].convert_tokens_to_ids(x) for x in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
    }
    load()
    organisms.load(S, key)
    for p in [p for p in shards("reads") if p != CACHE]:
        for line in p.open():
            r = json.loads(line)
            S["reads"][r["key"]] = r
            S["by_tid"].setdefault(r["tid"], {})[r["key"]] = r
    load_cache()  # this machine's reads, which follow_cache() keeps tailing
    S["pat_by_id"] = {p["pattern_id"]: p for p in S["patterns"]}


def main():
    global POD, JPP_LOCAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8803)
    ap.add_argument("--bind", nargs="+", default=os.environ.get("WCO_BIND", "127.0.0.1").split())
    ap.add_argument("--no-pod", action="store_true")
    ap.add_argument("--jpp-local", action="store_true", help="J++ reads go to a lens server on this machine (web/run_lens.sh --extras jpp)")
    ap.add_argument("--local", action="store_true", help="reads go to a lens server on this machine (web/run_lens.sh)")
    args = ap.parse_args()
    t0 = time.time()
    init()
    extras.load()
    steer.load()
    S["rubrics"] = {r["behavior_id"]: r["transcript_rubric"]["text"] for r in pq.read_table(HF / "rubrics.parquet").to_pylist()}
    if os.environ.get("WCO_JUDGE_URL"):
        steer.resume(judge_for, user_messages)
    from readq import ReadQueue

    S["queue"] = ReadQueue(ROOT / "data/queue.json", lambda k: k in S["reads"])
    threading.Thread(target=follow_cache, daemon=True).start()
    threading.Thread(target=consume, daemon=True).start()
    import sys

    sys.path.insert(0, str(HERE))
    if args.jpp_local:
        from local import Local

        JPP_LOCAL = Local()
    if args.local:
        from local import Local

        POD = Local()
    elif not args.no_pod:
        from pod import Pod

        POD = Pod()
    if POD:
        threading.Thread(target=autoprecompute, daemon=True).start()
    print(
        f"ready in {time.time() - t0:.0f}s: {len(S['patterns'])} patterns, {len(S['trs'])} transcripts, {len(S['reads'])} cached reads",
        flush=True,
    )
    for b in args.bind:
        s = (Server6 if ":" in b else ThreadingHTTPServer)((b, args.port), H)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        print(f"listening on {b}:{args.port}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
