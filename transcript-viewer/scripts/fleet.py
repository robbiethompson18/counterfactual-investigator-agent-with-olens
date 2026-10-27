"""Precompute lens reads on several GPU nodes in parallel, appending to data/reads.jsonl (the
viewer picks new lines up on its own).

Positions: every highlight transcript's end-of-turn zones, then user-turn punctuation, then
assistant-turn punctuation; at --layers. Reads already cached are skipped. Each node gets its own
cluster (wco-fleet-<i>), runs pod/fleet_job.sh as a sky job, and is downed as soon as the queue is
empty (the job also exits after 15 idle minutes, and the node autodowns 5 minutes later).

usage: .venv/bin/python scripts/fleet.py [--pods 4] [--layers 24,32,40,48,56] [--chunk 192]
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))
import server as V
from pod import hf_token

SKY = ["uvx", "--from", "skypilot[runpod,vast]==0.13.0", "sky"]
LOGDIR = ROOT / "outputs/fleet"
LOCK = threading.Lock()
STATS = {"done": 0, "t0": time.time()}


def log(name, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {name}: {msg}"
    print(line, flush=True)


def run(name, args, timeout=3600, env=None):
    with open(LOGDIR / f"{name}.log", "a") as f:
        f.write(f"\n[{time.strftime('%H:%M:%S')}] $ {' '.join(args)}\n")
        f.flush()
        return subprocess.run(args, check=False, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env=env, cwd=ROOT).returncode


def call(port, path, body=None, timeout=3600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(json.loads(e.read() or b"{}").get("error", f"HTTP {e.code}"))


def build_queue(layers, chunk, reverse=False):
    have = set()
    if V.CACHE.exists():
        for line in V.CACHE.read_text().splitlines():
            try:
                have.add(json.loads(line)["key"])
            except ValueError:
                pass
    todo = []
    for phase in ("zones", "user", "assistant"):
        for p in V.S["patterns"]:
            if not p["tids"]:
                continue
            tid = p["tids"][0]
            r = V.render(tid)
            if phase == "zones":
                pos = [i for a, b in r["zones"] for i in range(a, b)]
            else:
                a, b = r["msgs"][0 if phase == "user" else -1]
                pos = [i for i in range(a, b) if V.PUNCT.match(r["tokens"][i])]
            todo += [(tid, {"pos": i, "layer": L}) for i in pos for L in layers if V.key(tid, i, L) not in have]
    q = queue.Queue()
    chunks = [todo[k : k + chunk] for k in range(0, len(todo), chunk)]
    for c in chunks[::-1] if reverse else chunks:  # a second fleet works from the other end
        q.put(c)
    return q, len(todo)


def cluster_state(name):
    r = subprocess.run(SKY + ["status", name], check=False, capture_output=True, text=True, timeout=300, cwd=ROOT)
    for line in r.stdout.splitlines():
        p = line.split()
        if p and p[0] == name:
            for s in ("UP", "INIT", "STOPPED", "AUTOSTOPPING", "STOPPING"):
                if s in p:
                    return s
    return None


def server_running(name):
    r = subprocess.run(SKY + ["queue", name], check=False, capture_output=True, text=True, timeout=300, cwd=ROOT)
    # any live job counts: the multi-GPU tasks run their servers as the launch job itself
    return any(
        ("RUNNING" in ln or "PENDING" in ln or "SETTING_UP" in ln) and ("server" in ln or "fleet" in ln or "h200" in ln)
        for ln in r.stdout.splitlines()
    )


WEB = os.environ.get("WCO_VIEWER_URL", "http://127.0.0.1:8803")  # the viewer owns the queue (web/readq.py)
CHUNK = 192
IDLE_EXIT_S = 5 * 60


def web(path, body):
    req = urllib.request.Request(WEB + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def save(recs):
    with LOCK:
        with open(V.CACHE, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
        STATS["done"] += len(recs)


MIN_MIB = 80000  # Vast's "H100" offers come up as anything (32 GB RTX 5090s included)


KEEP = set()  # node indices to leave up when the queue is done (--keep)


def gpus(name):
    r = subprocess.run(  # noqa: PLW1510  (an unreachable node returns no output, handled below)
        [
            "ssh",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "LogLevel=ERROR",
            name,
            "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return [int(x) for x in r.stdout.split() if x.strip().isdigit()]


def node(i, q, total, env, first=0, yaml="sky/fleet.yaml"):
    name = f"wco-fleet-{i}"
    for attempt in range(4):
        ok = bring_up(name, i, first, env, yaml)
        if ok is None:
            return
        if ok:
            break
        log(name, "wrong GPU; tearing down and trying again")
        run(name, SKY + ["down", "-y", name], 900)
    else:
        return log(name, "no suitable GPU after 4 tries")
    n = len(gpus(name))
    log(name, f"{n} GPU(s); one worker each")
    ws = [threading.Thread(target=worker, args=(name, g, 18810 + 10 * i + g, q, total)) for g in range(n)]
    for w in ws:
        w.start()
    for w in ws:
        w.join()
    if i in KEEP:
        log(name, "queue done; keeping the node up (--keep)")
        return
    log(name, "tearing down")
    run(name, SKY + ["down", "-y", name], 900)


def bring_up(name, i, first, env, yaml):
    """True when the node has big-enough GPUs and its server job is running, False for the wrong
    GPU, None on failure."""
    time.sleep(min(30, 5 * abs(i - first)))
    st = cluster_state(name)
    if st is None:
        log(name, "launching")
        with_job = "fleet8" in yaml or "h200x" in yaml  # its run section is the server job
        args = ["launch", "-c", name, "-y", "--down", "-i", "5" if with_job else "30"]
        if with_job:
            args += ["-d", "--secret", "HF_TOKEN"]
        if run(name, SKY + args + [yaml], 3600, env) != 0:
            log(name, "launch failed (see outputs/fleet)")
            return None
        if with_job:
            log(name, f"GPUs: {gpus(name)} MiB; setup + servers running as the launch job")
            return True
    else:  # adopt a node an earlier driver launched
        log(name, f"adopting ({st})")
        t = time.time()
        while (st := cluster_state(name)) != "UP":
            if time.time() - t > 3600 or st in (None, "AUTOSTOPPING", "STOPPING"):
                log(name, f"adopted node not usable ({st})")
                if st is not None:
                    run(name, SKY + ["down", "-y", name], 900)
                return None
            time.sleep(20)
    mem = gpus(name)
    log(name, f"GPUs: {mem} MiB")
    if not mem or min(mem) < MIN_MIB:
        return False
    if not server_running(name):
        run(name, ["rsync", "-az", "pod/", f"{name}:sky_workdir/pod/"], 300)  # latest server code
        run(name, SKY + ["exec", name, "-d", "--name", "server", "--secret", "HF_TOKEN", "bash ~/sky_workdir/pod/fleet_job.sh"], 3600, env)
        log(name, "setup + server job started")
    else:
        log(name, "server job already running")
    if i in KEEP:
        run(name, SKY + ["autostop", "-y", name, "--cancel"], 300)
    else:
        run(name, SKY + ["autostop", "-y", name, "-i", "5", "--down"], 300)
    return True


def worker(node_name, g, port, q, total):
    name = f"{node_name}/gpu{g}"
    tunnel = None
    try:
        tunnel = subprocess.Popen(
            [
                "ssh",
                "-N",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ServerAliveInterval=30",
                "-L",
                f"{port}:127.0.0.1:{8802 + g}",
                node_name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        t = time.time()
        while True:
            try:
                if call(port, "/status", timeout=10).get("ready"):
                    break
            except Exception:  # noqa: BLE001  (server/worker boundary: report and keep going)
                if tunnel.poll() is not None:
                    tunnel = subprocess.Popen(
                        ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-L", f"{port}:127.0.0.1:{8802 + g}", node_name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            if time.time() - t > 3600:
                return log(name, "server not ready after 60 min, giving up")
            time.sleep(15)
        log(name, f"ready after {time.time() - t:.0f}s; reading")
        mem = gpus(node_name)
        # 80-85 GB cards run out of memory capturing long transcripts: leave those to the big ones
        max_tokens = 0 if mem and mem[g] > 90000 else 1536
        fails, n, idle = 0, CHUNK if not max_tokens else 48, None
        while True:
            try:
                items = web("/api/queue/take", {"n": n, "worker": name, "max_tokens": max_tokens})["items"]
            except Exception as e:  # noqa: BLE001  (server/worker boundary: report and keep going)
                log(name, f"viewer unreachable ({e}); retrying")
                time.sleep(30)
                continue
            if not items:  # nothing queued: wait a while for UI requests
                idle = idle or time.time()
                if time.time() - idle > IDLE_EXIT_S:
                    break
                time.sleep(15)
                continue
            idle = None
            chunk = [(it["tid"], {"pos": it["pos"], "layer": it["layer"]}) for it in items]
            keys = [it["key"] for it in items]
            jobs = {}
            for tid, it in chunk:
                jobs.setdefault(tid, []).append(it)
            try:
                t1 = time.time()
                res = call(port, "/read_multi", {"jobs": [{"ids": V.render(t)["ids"], "items": its} for t, its in jobs.items()]})
                fails = 0
            except Exception as e:  # noqa: BLE001  (server/worker boundary: report and keep going)
                web("/api/queue/release", {"keys": keys})
                if "out of memory" in str(e) and n > 12:  # 80-85 GB cards: smaller chunks
                    n //= 2
                    log(name, f"OOM at {len(chunk)} reads; chunks of {n} from now on")
                    continue
                fails += 1
                log(name, f"read failed ({type(e).__name__}: {e}); requeued")
                if fails >= 3:  # wait for the server to come back (restart, reload)
                    log(name, "3 failures in a row; waiting for the server to be ready again")
                    t2 = time.time()
                    while time.time() - t2 < 900:
                        try:
                            if call(port, "/status", timeout=10).get("ready"):
                                break
                        except Exception:  # noqa: BLE001  (server/worker boundary: report and keep going)
                            if tunnel.poll() is not None:
                                tunnel = subprocess.Popen(
                                    ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-L", f"{port}:127.0.0.1:{8802 + g}", node_name],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                )
                        time.sleep(15)
                    else:
                        return log(name, "server gone for 15 min, leaving the rest to the other nodes")
                    fails = 0
                time.sleep(20)
                continue
            recs = []
            for (tid, its), out, norms in zip(jobs.items(), res["results"], res["norms"]):
                toks = V.render(tid)["tokens"]
                recs += [
                    {
                        "key": V.key(tid, it["pos"], it["layer"]),
                        "tid": tid,
                        "pos": it["pos"],
                        "layer": it["layer"],
                        "temperature": 0.0,
                        "n": 1,
                        "seed": 0,
                        "token": toks[it["pos"]],
                        "norm": norms[j],
                        "samples": out[j],
                        "lens": V.pod_lens(res),
                        "t": time.time(),
                    }
                    for j, it in enumerate(its)
                ]
            save(recs)
            rate = STATS["done"] / max(1, time.time() - STATS["t0"]) * 60
            log(name, f"{len(recs)} reads in {time.time() - t1:.0f}s; total {STATS['done']}/{total} ({rate:.0f}/min overall)")
    finally:
        if tunnel:
            tunnel.terminate()
        log(name, "done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pods", type=int, default=4)
    ap.add_argument("--layers", default="24,32,40,48,56")
    ap.add_argument("--chunk", type=int, default=192)
    ap.add_argument("--first", type=int, default=0, help="first node index (a second fleet uses e.g. 4)")
    ap.add_argument("--reverse", action="store_true", help="work the queue from the end")
    ap.add_argument("--ids", default="", help="explicit node indices (adopt or launch), e.g. 1,8,9,20")
    ap.add_argument("--multi-ids", default="", help="which of --ids are multi-GPU nodes (sky/fleet8.yaml)")
    ap.add_argument("--keep", default="", help="node indices to leave up afterwards (autostop cancelled)")
    ap.add_argument("--yaml-for", default="", help="per-node task overrides, e.g. 22:sky/fleet_h200x2.yaml")
    args = ap.parse_args()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    V.S["tok"] = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    V.S["special"] = set(V.S["tok"].all_special_ids)
    V.load()
    global CHUNK
    CHUNK = args.chunk
    q, total = build_queue([int(x) for x in args.layers.split(",")], args.chunk, args.reverse)
    bulk = [{"tid": t, "pos": it["pos"], "layer": it["layer"]} for c in list(q.queue) for t, it in c]
    added = sum(web("/api/queue/bulk", {"items": bulk[k : k + 5000]})["added"] for k in range(0, len(bulk), 5000))
    log("fleet", f"{total} reads to do, {added} newly queued in the viewer's queue")
    env = dict(os.environ, HF_TOKEN=hf_token())
    ids = [int(x) for x in args.ids.split(",")] if args.ids else list(range(args.first, args.first + args.pods))
    multi = {int(x) for x in args.multi_ids.split(",") if x}
    KEEP.update(int(x) for x in args.keep.split(",") if x)
    custom = {int(k): v for k, v in (x.split(":") for x in args.yaml_for.split(",") if x)}
    ts = [
        threading.Thread(
            target=node,
            args=(
                i,
                q,
                total,
                env,
                ids[0] if args.ids else args.first,
                custom.get(i, "sky/fleet8.yaml" if i in multi else "sky/fleet.yaml"),
            ),
        )
        for i in ids
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    log("fleet", f"finished: {STATS['done']}/{total} reads; {q.qsize()} chunks left")


if __name__ == "__main__":
    main()
