"""Score Agam's O-lens readout banks with the AR: per-bullet and joint FVE for the 4 T=1 samples his s3d_rl600 lens left at
each token (data/banks/agam/<pid>/olens_bank_L{L}.json.gz), at each CHIVE incident's relevant positions (end-of-turn
zones + punctuation, server.relevant_positions) and all 11 layers. Agam decoded without a bullet cap, so 7% of samples
run on past 4 bullets; each is cut to its first 4 bullets, as our lens is (olens.lens.trim_bullets). Texts go to a lens
pod's POST /rescore (one capture per request, no decoding); records land in data/bank_scores/<pid>.jsonl in
data/reads.jsonl's shape. Only incidents with all 11 layers mirrored are scored; resumable (written keys are skipped).

usage: .venv/bin/python scripts/rescore_chive_banks.py [--url http://127.0.0.1:18803] [--limit N] [pid ...]
"""

import argparse
import gzip
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))
import chive
import server

LAYERS = chive.READERS[0]["layers"]
READER = "olens_bank"
CHUNK = 256  # items per /rescore request (each item: 4 samples, ~4 bullets each)
BULLET = re.compile(r"^\s*[-*•]", re.MULTILINE)  # olens.lens.BULLET (that module needs torch)


def trim(text, k=4):
    """olens.lens.trim_bullets: keep the first k bullet lines (plus anything before the first bullet)."""
    starts = [m.start() for m in BULLET.finditer(text)]
    return text[: starts[k]].rstrip() if len(starts) > k else text


def key(pid, pos, layer):
    return f"{pid}|{pos}|{layer}|1|4|bank"


def complete(pid):
    return all((chive.BANKS / pid / f"{READER}_L{L}.json.gz").exists() for L in LAYERS)


def post(url, body):
    req = urllib.request.Request(url + "/rescore", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def run(pid, url):
    """Score one incident; returns (records written, GPU seconds reported by the pod)."""
    out = chive.BANK_SCORES / f"{pid}.jsonl"
    done = {json.loads(line)["key"] for line in out.open()} if out.exists() else set()
    r = server.render(pid)
    positions = server.relevant_positions(pid)
    items = []
    for L in LAYERS:  # one bank file in memory at a time
        bank = json.loads(gzip.decompress((chive.BANKS / pid / f"{READER}_L{L}.json.gz").read_bytes()))
        assert len(bank) == len(r["ids"]), f"{pid} L{L}: bank has {len(bank)} cells for {len(r['ids'])} tokens"
        items += [{"pos": p, "layer": L, "texts": [trim(t) for t in bank[p]]} for p in positions if key(pid, p, L) not in done]
    n, secs = 0, 0.0
    for k in range(0, len(items), CHUNK):
        chunk = items[k : k + CHUNK]
        res = post(url, {"jobs": [{"ids": r["ids"], "items": chunk}]})
        recs = [
            {
                "key": key(pid, it["pos"], it["layer"]),
                "tid": pid,
                "pos": it["pos"],
                "layer": it["layer"],
                "temperature": 1.0,
                "n": 4,
                "seed": "bank",
                "bank": READER,
                "token": r["tokens"][it["pos"]],
                "norm": None,  # /rescore doesn't return ‖h‖
                "samples": samples,
                "t": time.time(),
            }
            for it, samples in zip(chunk, res["results"][0])
        ]
        with out.open("a") as f:
            f.writelines(json.dumps(x) + "\n" for x in recs)
        n += len(recs)
        secs += res["seconds"]
    return n, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pids", nargs="*", help="default: every incident with all 11 layers mirrored")
    ap.add_argument("--url", default="http://127.0.0.1:18803", help="the pod server, through an ssh tunnel")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many incidents")
    ap.add_argument("--workers", type=int, default=2, help="requests in flight (the pod runs them one at a time)")
    args = ap.parse_args()

    server.S["tok"] = server.AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    server.S["special"] = set(server.S["tok"].all_special_ids)
    server.load()
    chive.BANK_SCORES.mkdir(parents=True, exist_ok=True)
    pids = args.pids or sorted(chive.C["pats"])
    ready = [p for p in pids if complete(p)]
    print(f"{len(ready)}/{len(pids)} incidents have all {len(LAYERS)} {READER} layers", flush=True)
    if args.limit:
        ready = ready[: args.limit]

    t0, total = time.time(), 0
    with ThreadPoolExecutor(args.workers) as pool:
        for pid, (n, secs) in zip(ready, pool.map(lambda p: run(p, args.url), ready)):
            total += n
            rate = n / max(secs, 1e-9)
            print(f"{pid}: {n} records, pod {secs:.0f}s ({rate:.1f}/s); total {total} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
