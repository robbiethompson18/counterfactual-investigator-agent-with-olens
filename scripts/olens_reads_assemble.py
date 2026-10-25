"""Assemble a streamed O-lens read run (the parts scripts/olens_stream_upload.py put on HF) into reads.jsonl and finish it.

    uv run python scripts/olens_reads_assemble.py 2026-10-25_nocot-r4 [--requests REQ.jsonl] [--shards 8]
        [--status done|abandoned] [--missing MISSING.jsonl] [--no-finish]

Downloads runs/olens_reads/RUN/{parts,requests,logs}/ (and meta.json, if the local run dir lacks it) from the HF dataset
into data/runs/olens_reads/RUN, concatenates the parts into reads.jsonl, writes requests.jsonl (--requests, else the union
of the pods' uploaded requests files), reports completeness per shard (requested vs read ids; shard = zlib.crc32(id) % n as
the reader's queue()), and runs scripts/runs.py finish(): status + sync, so the run on HF has the reads.jsonl every
consumer expects (cfbench --olens-reads). Idempotent: rerun it after more parts land.

A transcript in two parts (e.g. two pods run without --shard) is taken from the first part by name; lines that don't parse
(torn by a kill mid-write) are dropped and counted. --missing writes the requests with no reads, ready to re-queue.
"""

import argparse
import json
import shutil
import zlib
from pathlib import Path

from huggingface_hub import snapshot_download
from runs import REPO, ROOT, finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run dir name (or path) under data/runs/olens_reads/")
    ap.add_argument("--requests", type=Path, help="the requests file the pods read; default: the union of their uploaded copies")
    ap.add_argument("--shards", type=int, help="number of shards; default: the number of distinct tags among the parts")
    ap.add_argument("--status", default="done", choices=["done", "abandoned"])
    ap.add_argument("--missing", type=Path, help="write the requests that have no reads here")
    ap.add_argument("--no-finish", action="store_true", help="assemble and report only; don't touch meta.json or sync")
    args = ap.parse_args()
    run = ROOT / "olens_reads" / Path(args.run).name
    prefix = f"runs/olens_reads/{run.name}"
    patterns = [f"{prefix}/{d}/*" for d in ("parts", "requests", "logs")]
    if not (run / "meta.json").exists():  # never overwrite a local meta.json with HF's older copy
        patterns.append(f"{prefix}/meta.json")
    snapshot_download(REPO, repo_type="dataset", local_dir=ROOT.parent, allow_patterns=patterns)
    if not (run / "meta.json").exists():
        raise SystemExit(f"{run}/meta.json is neither local nor on HF: create the run with scripts/runs.py new olens_reads SLUG ...")

    parts = sorted((run / "parts").glob("*.jsonl"))
    if not parts:
        raise SystemExit(f"no parts under {prefix}/parts/ on HF")
    owner: dict[str, str] = {}  # id -> the part it's taken from
    torn = 0
    for part in parts:
        for line in part.open():
            try:
                owner.setdefault(json.loads(line)["id"], part.name)
            except json.JSONDecodeError:
                torn += 1
    rows, dup_ids = 0, set()
    with (run / "reads.jsonl").open("w") as f:
        for part in parts:
            for line in part.open():
                try:
                    rid = json.loads(line)["id"]
                except json.JSONDecodeError:
                    continue
                if owner[rid] == part.name:
                    f.write(line if line.endswith("\n") else line + "\n")
                    rows += 1
                else:
                    dup_ids.add(rid)

    if args.requests:
        shutil.copyfile(args.requests, run / "requests.jsonl")
    else:
        uploaded = sorted((run / "requests").glob("*.jsonl"))
        if not uploaded:
            raise SystemExit("no requests files were uploaded; pass --requests")
        seen: set[str] = set()
        with (run / "requests.jsonl").open("w") as f:
            for path in uploaded:
                for line in path.open():
                    rid = json.loads(line)["id"]
                    if rid not in seen:
                        seen.add(rid)
                        f.write(line)
    requested = [json.loads(line)["id"] for line in (run / "requests.jsonl").open() if line.strip()]

    tags = {p.name.split(".reads")[0] for p in parts}
    n = args.shards or len(tags)
    print(f"{rows} rows, {len(owner)} transcripts from {len(parts)} parts ({len(tags)} tags), {torn} torn lines dropped, "
          f"{len(dup_ids)} transcripts in more than one part (first copy kept)")  # fmt: skip
    missing = {rid for rid in requested if rid not in owner}
    for i in range(n):
        want = [rid for rid in requested if zlib.crc32(rid.encode()) % n == i]
        got = sum(rid in owner for rid in want)
        print(f"  shard {i}/{n}: {got}/{len(want)} read" + (f", {len(want) - got} missing" if got < len(want) else ""))
    stray = len(set(owner) - set(requested))
    print(f"total: {len(requested) - len(missing)}/{len(requested)} requested transcripts read"
          + (f", {stray} read but not requested" if stray else ""))  # fmt: skip
    if args.missing:
        with args.missing.open("w") as f:
            for line in (run / "requests.jsonl").open():
                if json.loads(line)["id"] in missing:
                    f.write(line)
        print(f"wrote {len(missing)} unread requests to {args.missing}")

    if not args.no_finish:
        meta_path = run / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["reads"] = {"rows": rows, "transcripts": len(owner), "requested": len(requested), "missing": len(missing)}
        meta["reads"] |= {"parts": len(parts), "torn_lines": torn}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        finish(run, args.status)


if __name__ == "__main__":
    main()
