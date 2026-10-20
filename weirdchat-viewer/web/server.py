"""WeirdChat x oracle lens viewer: http://<host>:8803/ (binds to $WCO_BIND, default 127.0.0.1)

Browses the Qwen3.6-27B patterns of Transluce/WeirdChat with their transcripts rendered through the
model's chat template, token by token. Clicking a token reads it with the agu18dec oracle lens
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

import pyarrow.parquet as pq
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HF = ROOT / "data/hf/data"
CACHE = ROOT / "data/reads.jsonl"
CLAUDE = ROOT / "data/claude.jsonl"  # scripts/claude_rollouts.py: Haiku on the same prompts
STANCE = ROOT / "data/stance.jsonl"  # per-tid ENDORSE/CONDITIONAL/AGAINST from a Sonnet 5.5 stance classifier (3 votes)
MODEL = "qwen/qwen3.6-27b"
LAYERS = list(range(20, 61, 4))
MAX_TOKENS = 4096
PRE_LAYERS = [40]  # precomputed on every highlight transcript (see precompute())
PRE_BATCH = 96
PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")
# GPT-2 byte-level BPE alphabet (bytes_to_unicode, inverted): token char -> raw byte
_KEEP = [*range(33, 127), *range(161, 173), *range(174, 256)]
BYTE = {chr(b): b for b in _KEEP} | {chr(256 + n): b for n, b in enumerate(b for b in range(256) if b not in _KEEP)}

S = {"reads": {}, "by_tid": {}, "pre": {"state": "idle", "done": 0, "total": 0}}
LOCK = threading.Lock()
POD = None


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
    S.update(patterns=pats, prompts=prompts, trs=trs)


def render(tid):
    """Chat-template the transcript, tokenise with offsets, and locate messages, citations and
    the end-of-turn zones (<|im_end|> and the template tokens up to the next content)."""
    if tid in S.setdefault("rendered", {}):
        return S["rendered"][tid]
    tok, t = S["tok"], S["trs"][tid]
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
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    zones = []  # [start, end) token ranges: an <|im_end|> through the next content
    starts = [r[0] for r in msg_tok if r] + [len(ids)]
    for i, x in enumerate(ids):
        if x == im_end:
            nxt = min([s for s in starts if s > i] or [len(ids)])
            zones.append([i, min(nxt, len(ids))])
    # byte-level BPE splits multi-byte characters (emoji) across tokens, and decoding one id alone turns each
    # piece into U+FFFD: decode the raw bytes incrementally, so a character shows on the token holding its last byte
    utf8 = codecs.getincrementaldecoder("utf-8")("replace")
    toks = [utf8.decode(bytes(BYTE[c] for c in piece)) for piece in tok.convert_ids_to_tokens(ids)]
    out = {
        "tid": tid,
        "ids": ids,
        "tokens": toks,
        "special": [x in S["special"] for x in ids],
        "msgs": msg_tok,
        "roles": [m["role"] for m in msgs],
        "cites": cites,
        "zones": zones,
        "truncated": len(enc.input_ids) > MAX_TOKENS,
    }
    S["rendered"][tid] = out
    return out


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


def overview():
    """Per pattern: the cached greedy L40 reads in the highlight transcript's first end-of-turn
    zone (after the user's message), for the list."""
    out = {}
    for p in S["patterns"]:
        if not p["tids"]:
            continue
        tid = p["tids"][0]
        z = render(tid)["zones"][:1]
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
                    "coefs": r["samples"][0]["coefs"][:4],
                    "bfve": r["samples"][0].get("bfve", [])[:4],
                    "fve": r["samples"][0]["fve"],
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
            if not p["tids"]:
                continue
            tid = p["tids"][0]
            r = render(tid)
            pos = [i for a, b in r["zones"] for i in range(a, b)] if phase == "zones" else punct_positions(r)
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
    """End-of-turn zones + punctuation in both turns: where the lens reads best."""
    r = render(tid)
    return sorted({i for a, b in r["zones"] for i in range(a, b)} | set(punct_positions(r, 0)) | set(punct_positions(r, -1)))


def queue_items(tid, positions, layers, src):
    toks = render(tid)["tokens"]
    return [{"key": key(tid, p, L), "tid": tid, "pos": p, "layer": L, "src": src} for p in positions for L in layers if 0 <= p < len(toks)]


def title_of(tid):
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
    e = p["elo"] or {}
    return {
        "idx": p["idx"],
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
    }


def pattern_detail(i):
    p = S["patterns"][i]
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
    )


def claude_rollouts(prompt_id):
    """Claude rollouts for a prompt, re-reading data/claude.jsonl when it changes."""
    if not CLAUDE.exists():
        return []
    mtime = CLAUDE.stat().st_mtime
    if S.get("claude_mtime") != mtime:
        by_prompt = {}
        for line in CLAUDE.open():
            r = json.loads(line)
            by_prompt.setdefault(r["prompt_id"], []).append(r)
        S.update(claude=by_prompt, claude_mtime=mtime)
    return S["claude"].get(prompt_id, [])


def stance(tid):
    """Stance label for a transcript, re-reading data/stance.jsonl when it changes."""
    if not STANCE.exists():
        return None
    mtime = STANCE.stat().st_mtime
    if S.get("stance_mtime") != mtime:
        S.update(stance={r["tid"]: r["stance"] for r in map(json.loads, STANCE.open())}, stance_mtime=mtime)
    return S["stance"].get(tid)


def transcript(tid):
    r = render(tid)
    t = S["trs"][tid]
    out = {k: v for k, v in r.items() if k != "ids"}
    out.update(
        queued=[[int(k.split("|")[1]), int(k.split("|")[2])] for k in S["queue"].pending_keys(tid)],
        relevant=len(relevant_positions(tid)),
        punct=punct_positions(r, 0) + punct_positions(r, -1),
        user_punct=len(punct_positions(r, 0)),
        match=t["judgment"]["match"],
        explanation=t["judgment"]["explanation"],
        highlight=t["is_highlight"],
        url=t["weirdchat_url"],
        reads=tid_reads(tid),
        claude=claude_rollouts(t["prompt_id"]),
        stance=stance(tid),
    )
    return out


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
                t = S["trs"][q["tid"]]
                return self.send({"pattern": S["pat_by_id"][t["pattern_id"]]["idx"]})
            if u.path == "/api/queue":
                return self.send(S["queue"].summary(title_of))
            if u.path == "/api/status":
                return self.send(
                    {
                        "pod": POD.status() if POD else {"state": "disabled"},
                        "pre": S["pre"],
                        "n_reads": len(S["reads"]),
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
            if p == "/api/pod/start" and POD:
                return self.send(dict(started=POD.start(), **POD.status()))
            if p == "/api/pod/stop" and POD:
                return self.send(dict(stopped=POD.stop(), **POD.status()))
            if p == "/api/queue/add":  # from the UI: to the front
                tid = req["tid"]
                layers = [int(L) for L in req.get("layers") or QUEUE_LAYERS if int(L) in LAYERS]
                pos = relevant_positions(tid) if req.get("scope", "relevant") == "relevant" else [int(x) for x in req["positions"]]
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


def main():
    global POD
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8803)
    ap.add_argument("--bind", nargs="+", default=os.environ.get("WCO_BIND", "127.0.0.1").split())
    ap.add_argument("--no-pod", action="store_true")
    ap.add_argument("--local", action="store_true", help="reads go to a lens server on this machine (web/run_lens.sh)")
    args = ap.parse_args()
    t0 = time.time()
    S["tok"] = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    S["special"] = set(S["tok"].all_special_ids) | {
        S["tok"].convert_tokens_to_ids(x) for x in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
    }
    load()
    load_cache()
    from readq import ReadQueue

    S["pat_by_id"] = {p["pattern_id"]: p for p in S["patterns"]}
    S["queue"] = ReadQueue(ROOT / "data/queue.json", lambda k: k in S["reads"])
    threading.Thread(target=follow_cache, daemon=True).start()
    threading.Thread(target=consume, daemon=True).start()
    import sys

    sys.path.insert(0, str(HERE))
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
