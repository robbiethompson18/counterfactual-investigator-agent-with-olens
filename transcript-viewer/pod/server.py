"""Lens read server on a GPU node (bound to 127.0.0.1; the viewer reaches it over an ssh tunnel).

One Qwen3.6-27B (HF, bf16) serves activation capture, the agu18dec lens and its AR, all as
hook-LoRAs from ~/olens/src/olens. GPU work runs on the main thread (other threads were 2-5x
slower on CUDA, see olens commit c2599c1).

GET  /status      {ready, busy, task, up, idle}
POST /extras      {ids, positions, nla, jpp} -> {nla: [{text, cos, fve}], jpp: {pos: [[[tok, p]] * k per layer]}}
                  the ceselder L42 NLA and the J++ lens (olens.extras); loaded on >= 120 GB GPUs or --extras
POST /vector      {spec} -> {v, layer, info}            probe/steering direction (olens.vectors)
POST /probe       {ids, layer, v} -> {cos, proj}       per position
POST /steer       {ids, layer, v, alpha, n, max_new_tokens, temperature, seed} -> {texts}
POST /contrib     {ids, pos, vecs} -> per block: attention / MLP write norms and cosines
POST /read        {ids, items: [{pos, layer}], temperature, n, seed, max_bullets}
                  -> {results: [[{text, bullets, coefs, fve}] * n per item], seconds}
                  one capture for the whole sequence, every (pos, layer) decoded in batches.

Exits after --idle-min minutes without a read, or after --max-hours, so the sky job ends and
SkyPilot's autodown takes the pod down even if the viewer's host or the tunnel are gone.
"""

import argparse
import gc
import json
import os
import queue
import threading
import time
import traceback
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from olens.ar import AR
from olens.fve import fve_nnls, fve_single
from olens.lens import AguLens, Sampling, capture, load_base
from olens.readout import Readout, bullets

S = {}
JOBS = queue.Queue()
STATUS = {"ready": False, "busy": False, "task": None, "since": None}
T0 = time.time()
LAST = {"t": time.time()}
BATCH = 96


def gpu(task, fn):
    fut = Future()

    def job():
        STATUS.update(busy=True, task=task, since=time.time())
        err = None
        try:
            fut.set_result(fn())
        except BaseException as e:  # noqa: BLE001  (server/worker boundary: report and keep going)
            # pass on the message only: the traceback's frames hold the GPU tensors alive
            err = f"{type(e).__name__}: {e}"
        finally:
            STATUS.update(busy=False, task=None, since=None)
        if err:  # outside the except block, so the traceback is gone
            gc.collect()
            torch.cuda.empty_cache()
            fut.set_exception(RuntimeError(err))

    JOBS.put(job)
    return fut.result()


@torch.no_grad()
def read_many(vecs, layers, smp, n=1, seed=0):
    """vecs [B, d], layers [B] -> [B][n] scored readouts. The carrier prompts for layers 20..60
    differ only in the layer digits, so every row has the same length and marker slot."""
    lens, _ro = S["agu"], S["readout"]
    texts = []
    for i in range(0, len(layers), max(1, BATCH // n)):
        L, v = layers[i : i + BATCH // n], vecs[i : i + BATCH // n]
        carriers = [lens._carrier(l) for l in L]
        assert len({(len(c), s) for c, s in carriers}) == 1
        ids = [c for c, _ in carriers for _ in range(n)]
        torch.manual_seed(seed)
        with lens.lora:
            out = lens._generate(ids, carriers[0][1], v.to(lens.model.device).float().repeat_interleave(n, 0), smp)
        dec = [lens._finish(t, smp) for t in lens.tok.batch_decode(out, skip_special_tokens=True)]
        texts += [dec[j * n : (j + 1) * n] for j in range(len(L))]
    return score(texts, vecs, layers)


@torch.no_grad()
def score(texts, vecs, layers):
    """texts [B][n] readouts of vecs [B, d] at layers [B] -> scored readouts. Each bullet goes
    through the AR; `fve` is the joint whitened FVE of the NNLS refit on all bullets, `bfve` each
    bullet's own FVE (cos^2 in the whitened space, 0 if the cosine is negative), `coefs` the NNLS
    weights. Bullets are sorted by their own FVE."""
    lens, ro = S["agu"], S["readout"]
    flat = [(b, k, L) for k, (ts, L) in enumerate(zip(texts, layers)) for t in ts for b in bullets(t)]
    spans = [lens.tok(b, add_special_tokens=False).input_ids[:32] for b, _, _ in flat]
    D = torch.cat([ro.ar(spans[i : i + 256], [L for _, _, L in flat[i : i + 256]]) for i in range(0, len(spans), 256)]) if spans else None
    out, j = [], 0
    for k, (ts, L) in enumerate(zip(texts, layers)):
        W = ro.whitener(L)
        row = []
        for t in ts:
            bl = bullets(t)
            if not bl:
                row.append({"text": t, "bullets": [], "coefs": [], "bfve": [], "fve": 0.0})
                continue
            z, Dz = W(vecs[k].to(D.device)), W(D[j : j + len(bl)])
            j += len(bl)
            fve, c = fve_nnls(z, Dz)
            single, _ = fve_single(z[None], Dz)
            order = sorted(range(len(bl)), key=lambda i: -float(single[i]))
            row.append(
                {
                    "text": t,
                    "bullets": [bl[i] for i in order],
                    "coefs": [round(float(c[i]), 4) for i in order],
                    "bfve": [round(float(single[i]), 4) for i in order],
                    "fve": round(float(fve), 4),
                }
            )
        out.append(row)
    return out


def do_rescore(req):
    """{jobs: [{ids, items: [{pos, layer, texts}]}]} -> {results: [per job: [per item: samples]]}:
    re-score cached readout texts (adds per-bullet FVE) without decoding again."""
    t0 = time.time()
    vecs, layers, texts, sizes = [], [], [], []
    for job in req["jobs"]:
        ids = [int(i) for i in job["ids"]]
        acts = capture(S["model"], ids, sorted({int(it["layer"]) for it in job["items"]}))
        for it in job["items"]:
            vecs.append(acts[int(it["layer"])][int(it["pos"]) % len(ids)].cpu())
            layers.append(int(it["layer"]))
            texts.append(it["texts"])
        sizes.append(len(job["items"]))
        del acts
    res = score(texts, torch.stack(vecs), layers)
    out, i = [], 0
    for k in sizes:
        out.append(res[i : i + k])
        i += k
    return {"results": out, "seconds": round(time.time() - t0, 1)}


def do_read(req):
    ids = [int(i) for i in req["ids"]]
    items = req["items"]
    need = sorted({int(it["layer"]) for it in items})
    assert all(l in AguLens.layers for l in need), f"layers must be in {AguLens.layers}"
    t0 = time.time()
    acts = capture(S["model"], ids, need)
    vecs = torch.stack([acts[int(it["layer"])][int(it["pos"]) % len(ids)] for it in items]).cpu()
    del acts
    smp = Sampling(
        temperature=float(req.get("temperature", 0)),
        max_new_tokens=int(req.get("max_new_tokens", 160)),
        max_bullets=int(req.get("max_bullets", 4)) or None,
    )
    res = read_many(vecs, [int(it["layer"]) for it in items], smp, int(req.get("n", 1)), int(req.get("seed", 0)))
    return {"results": res, "norms": [round(float(v.norm()), 1) for v in vecs], "seconds": round(time.time() - t0, 1)}


def do_read_multi(req):
    """{jobs: [{ids, items}], ...sampling} -> {results: [per job: [per item]], norms}: one capture per
    sequence, then every item of every job decoded together in full batches."""
    t0 = time.time()
    vecs, layers, sizes = [], [], []
    for job in req["jobs"]:
        ids = [int(i) for i in job["ids"]]
        need = sorted({int(it["layer"]) for it in job["items"]})
        acts = capture(S["model"], ids, need)
        vecs += [acts[int(it["layer"])][int(it["pos"]) % len(ids)].cpu() for it in job["items"]]
        layers += [int(it["layer"]) for it in job["items"]]
        sizes.append(len(job["items"]))
        del acts
    smp = Sampling(
        temperature=float(req.get("temperature", 0)),
        max_new_tokens=int(req.get("max_new_tokens", 160)),
        max_bullets=int(req.get("max_bullets", 4)) or None,
    )
    V = torch.stack(vecs)
    res = read_many(V, layers, smp, int(req.get("n", 1)), int(req.get("seed", 0)))
    out, norms, i = [], [], 0
    for k in sizes:
        out.append(res[i : i + k])
        norms.append([round(float(v.norm()), 1) for v in V[i : i + k]])
        i += k
    return {"results": out, "norms": norms, "seconds": round(time.time() - t0, 1)}


def do_extras(req):
    ids, pos = [int(i) for i in req["ids"]], [int(p) % len(req["ids"]) for p in req["positions"]]
    if "jpp" not in S:
        raise RuntimeError("NLA/J++ not loaded on this pod (needs a >= 120 GB GPU or --extras)")
    t0, out = time.time(), {}
    if req.get("jpp", True):
        out["jpp"] = {str(p): v for p, v in S["jpp"].read(ids, pos, k=int(req.get("k", 10))).items()}
        out["jpp_layers"] = S["jpp"].layers
    if req.get("nla", True) and "nla" in S:
        v = capture(S["model"], ids, [42])[42][pos].cpu()
        texts = S["nla"].explain(v, temperature=float(req.get("temperature", 0)))
        out["nla"] = [dict(text=t, **sc) for t, sc in zip(texts, S["critic"].score(texts, v))]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def load_extras(nla=True):
    from olens.extras import JPP, NLA, NLACritic

    S["jpp"] = JPP(S["model"])
    if not nla:
        return print(f"J++ loaded {time.time() - T0:.0f}s", flush=True)
    S["nla"] = NLA(S["tok"], S["model"], stack=os.environ.get("NLA_STACK", "1") == "1")  # RL LoRA sits on the SFT-merged base
    S["critic"] = NLACritic(S["tok"])
    calib = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_ids.json")))["ids"]
    S["critic"].calibrate(torch.cat([capture(S["model"], ids, [42])[42][4:] for ids in calib]))
    print(f"NLA + J++ loaded {time.time() - T0:.0f}s (FVE baseline {S['critic'].base:.1f})", flush=True)


def handler(name):
    """pod/handlers.py's `name`, freshly reloaded together with olens.vectors (no model reload)."""
    import importlib
    import sys

    import olens.vectors

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import handlers

    importlib.reload(olens.vectors)
    importlib.reload(handlers)
    handlers.S = S
    return getattr(handlers, name)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/status"):
            st = dict(
                STATUS,
                up=int(time.time() - T0),
                idle=int(time.time() - LAST["t"]),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            )
            st["elapsed"] = round(time.time() - st["since"], 1) if st["since"] else None
            return self.send(st)
        self.send({"error": "not found"}, 404)

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/read":
                if not STATUS["ready"]:
                    return self.send({"error": "model still loading"}, 503)
                LAST["t"] = time.time()
                r = gpu(f"{len(req['items'])} reads", lambda: do_read(req))
                LAST["t"] = time.time()
                return self.send(r)
            if self.path == "/rescore":
                if not STATUS["ready"]:
                    return self.send({"error": "model still loading"}, 503)
                LAST["t"] = time.time()
                n = sum(len(j["items"]) for j in req["jobs"])
                r = gpu(f"re-scoring {n} reads", lambda: do_rescore(req))
                LAST["t"] = time.time()
                return self.send(r)
            if self.path in ("/vector", "/probe", "/steer", "/nla_score", "/contrib") or self.path.startswith("/h/"):
                if not STATUS["ready"]:
                    return self.send({"error": "model still loading"}, 503)
                # /h/<name> -> handlers.do_<name>, so new handlers need no server restart either
                name = "do_" + self.path[3:] if self.path.startswith("/h/") else {"/vector": "do_vector", "/probe": "do_probe", "/steer": "do_steer", "/nla_score": "do_nla_score", "/contrib": "do_contrib"}[self.path]
                fn = handler(name)
                LAST["t"] = time.time()
                r = gpu(self.path[1:], lambda: fn(req))
                LAST["t"] = time.time()
                return self.send(r)
            if self.path == "/extras":
                if not STATUS["ready"]:
                    return self.send({"error": "model still loading"}, 503)
                LAST["t"] = time.time()
                r = gpu(f"NLA/J++ on {len(req['positions'])} tokens", lambda: do_extras(req))
                LAST["t"] = time.time()
                return self.send(r)
            if self.path == "/read_multi":
                if not STATUS["ready"]:
                    return self.send({"error": "model still loading"}, 503)
                LAST["t"] = time.time()
                n = sum(len(j["items"]) for j in req["jobs"])
                r = gpu(f"{n} precompute reads", lambda: do_read_multi(req))
                LAST["t"] = time.time()
                return self.send(r)
            self.send({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001  (server/worker boundary: report and keep going)
            traceback.print_exc()
            self.send({"error": f"{type(e).__name__}: {e}"}, 500)


def reaper(idle_min, max_hours):
    while True:
        time.sleep(20)
        if STATUS["busy"]:
            continue
        idle = time.time() - LAST["t"]
        if (idle_min and idle > idle_min * 60) or (max_hours and time.time() - T0 > max_hours * 3600):
            print(f"exiting: idle {idle / 60:.0f} min, up {(time.time() - T0) / 3600:.2f} h", flush=True)
            os._exit(0)


def main():
    global BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8802)
    ap.add_argument("--idle-min", type=float, default=45, help="0 = never exit on idle")
    ap.add_argument("--max-hours", type=float, default=3.75, help="0 = no cap (local GPU)")
    ap.add_argument("--batch", type=int, default=BATCH, help="decode batch (reads x samples)")
    ap.add_argument("--extras", choices=["auto", "on", "jpp", "off"], default="auto", help="NLA + J++ (auto: on >= 120 GB GPUs); jpp = J++ only")
    ap.add_argument("--no-warmup", action="store_true", help="skip the CUDA kernel warm-up (ROCm doesn't need it)")
    args = ap.parse_args()
    BATCH = args.batch
    threading.Thread(target=reaper, args=(args.idle_min, args.max_hours), daemon=True).start()
    s = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    print(f"listening on 127.0.0.1:{args.port}", flush=True)

    tok, model = load_base()
    agu = AguLens(tok, model)
    S.update(tok=tok, model=model, agu=agu, readout=Readout(agu, AR(model)))
    for L in AguLens.layers:
        S["readout"].whitener(L)
    big = torch.cuda.get_device_properties(0).total_memory > 120e9
    if args.extras == "on" or (args.extras == "auto" and big):
        load_extras()
    elif args.extras == "jpp":
        load_extras(nla=False)
    print(f"loaded {time.time() - T0:.0f}s", flush=True)
    # Warm-up: the linear-attention kernels compile per sequence length on first sight; walk the
    # lengths a read reaches for the batch sizes the viewer uses (1 and 11 live, full batches
    # for precompute). Persisted with TRITON_CACHE_AUTOTUNING.
    STATUS["task"] = "warming up kernels"
    for b in () if args.no_warmup else (1, 11, BATCH):
        read_many(torch.randn(b, 5120), [40] * b, Sampling(max_new_tokens=170, min_new_tokens=170))
        print(f"warm-up batch {b} done {time.time() - T0:.0f}s", flush=True)
    STATUS.update(ready=True, task=None)
    LAST["t"] = time.time()
    print(f"ready {time.time() - T0:.0f}s", flush=True)
    while True:
        JOBS.get()()


if __name__ == "__main__":
    main()
