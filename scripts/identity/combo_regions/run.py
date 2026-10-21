"""Sample every (base, region) cell x probe, thinking on, regex-label, append to samples.jsonl as requests finish,
and one progress.tsv line per finished cell.

    uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/combo_regions/run.py \
        --bases none,A,B --regions aws_usw2,gcp_usw1 --n 32
"""

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/combo_regions")
from cues import BASES, PROBES, REGIONS, cot_flags, msgs, score
from openai import AsyncOpenAI
from qwen import VLLMS, _sample, _up

OUT = Path("data/runs/identity/2026-10-21_combo_regions")
CHUNK = 8


def done_counts():
    c = Counter()
    if (OUT / "samples.jsonl").exists():
        for line in (OUT / "samples.jsonl").open():
            r = json.loads(line)
            c[(r["base"], r["region"], r["probe"])] += 1
    return c


def cell_line(base, region, rows):
    by = defaultdict(list)
    for r in rows:
        by[r["probe"]].append(r)
    cols = []
    for p in PROBES:
        rs = by[p]
        cols.append(json.dumps(dict(Counter(r["label"] for r in rs).most_common(4))))
    cot = sum(r["cot_creator"] for r in rows), sum(r["cot_cloud"] for r in rows), len(rows)
    return "\t".join([base, region, str(len(rows))] + cols + [f"{cot[0]}/{cot[2]}", f"{cot[1]}/{cot[2]}"]) + "\n"


async def main(cells, n, max_tokens, concurrency):
    have = done_counts()
    live = [u for u in VLLMS if _up(u)]
    clients = [AsyncOpenAI(base_url=u, api_key="none", max_retries=2, timeout=1800) for u in live]
    sem = asyncio.Semaphore(concurrency)
    jobs = [(b, r, p, k) for b, r in cells for p in PROBES for k in range(max(0, n - have[(b, r, p)]) // CHUNK)]
    remaining = Counter((b, r) for b, r, _, _ in jobs)
    cell_rows = defaultdict(list)
    print(f"{len(jobs)} requests x {CHUNK} over {len(live)} replicas", flush=True)
    f = (OUT / "samples.jsonl").open("a")

    async def one(i, b, r, p, k):
        ss = await _sample(clients[i % len(clients)], sem, msgs(b, r, PROBES[p]), CHUNK, True, None, max_tokens)
        return b, r, p, k, ss

    tasks = [asyncio.create_task(one(i, *j)) for i, j in enumerate(jobs)]
    for t in asyncio.as_completed(tasks):
        try:
            b, r, p, k, ss = await t
        except Exception as e:  # noqa: BLE001  (sampler loop: log and keep going; a rerun fills the gap)
            print("ERR", repr(e)[:200], flush=True)
            continue
        rows = [
            {
                "base": b,
                "region": r,
                "probe": p,
                "i": k * CHUNK + j,
                **s,
                "label": score(p, s["content"]) if s["finish_reason"] == "stop" else "trunc",
                **cot_flags(s["reasoning"]),
            }
            for j, s in enumerate(ss)
        ]
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
        f.flush()
        cell_rows[(b, r)] += rows
        remaining[(b, r)] -= 1
        if remaining[(b, r)] == 0:
            with (OUT / "progress.tsv").open("a") as pf:
                pf.write(cell_line(b, r, cell_rows[(b, r)]))
            print("cell done", b, r, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default=",".join(BASES))
    ap.add_argument("--regions", default=",".join(REGIONS))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=128)
    a = ap.parse_args()
    cells = [(b, r) for r in a.regions.split(",") for b in a.bases.split(",")]
    pt = OUT / "progress.tsv"
    if not pt.exists():
        pt.write_text("\t".join(["base", "region", "n_rows"] + list(PROBES) + ["cot_creator", "cot_cloud"]) + "\n")
    asyncio.run(main(cells, a.n, a.max_tokens, a.concurrency))
