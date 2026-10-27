"""A batch-reads run as an expview dataset, without copying the reads (docs/olens-reads-format.md).

    python expview/importers/reads.py RUN_DIR [RUN_DIR ...]

For every X.transcripts.jsonl in RUN_DIR with its reads X.jsonl beside it (scripts/olens_read_requests.py and
jpp_read_requests.py write both), writes an expview run at RUN_DIR/expview/ (X = reads) or RUN_DIR/expview/X/: one item per
transcript that has reads, whose `tokens` entry points at byte ranges of the two files instead of a tokens/ file. The page
fetches those ranges from HF and builds the transcript with its reads in the browser. `scripts/runs.py finish` runs this, so
a reads run is viewable once it is synced. Rerunning rewrites the items. An expview/ dir this didn't write is left alone.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from expview import Run  # ty: ignore[unresolved-import]

ID = re.compile(rb'^\{"id": "((?:[^"\\]|\\.)*)"')  # every reader writes `id` first; anything else is parsed whole
POS = re.compile(rb'"token_index": (\d+)')
EPISODE = re.compile(r"^[^/:|]+/[^/:|]+:\d+")  # ENV/RUN:LINE, the evalcue-style rollout id (any |suffix dropped)


def line_id(line: bytes) -> str:
    m = ID.match(line)
    return json.loads(b'"' + m.group(1) + b'"') if m else json.loads(line)["id"]


def spans(path: Path) -> dict[str, list]:
    """id -> ([[offset, length], ...] of its runs of consecutive lines, set of token positions). Readers append a transcript's
    rows in one block, but merged shards or a resumed run can split it."""
    out, off, cur = {}, 0, None
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                i = line_id(line)
                sp, pos = out.setdefault(i, ([], set()))
                if i == cur:
                    sp[-1][1] += len(line)
                else:
                    sp.append([off, len(line)])
                m = POS.search(line)
                if m:
                    pos.add(int(m.group(1)))
                cur = i
            off += len(line)
    return out


def transcripts(path: Path) -> dict[str, tuple[int, int, dict]]:
    """id -> (offset, length, metadata) of its last line (a reread after a crash may repeat one)."""
    out, off = {}, 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                meta = {k: v for k, v in t.items() if k not in ("tokens", "sections")}
                out[t["id"]] = (off, len(line), {**meta, "n": len(t["tokens"])})
            off += len(line)
    return out


def import_reads(run_dir: Path, reads: Path, tpath: Path) -> Path | None:
    name = reads.name.removesuffix(".jsonl")
    out = run_dir / "expview" if name == "reads" else run_dir / "expview" / name
    rj = out / "run.json"
    if (out / "items.jsonl").exists() and not (rj.exists() and json.loads(rj.read_text()).get("reads_file")):
        print(f"{out}: an expview run not made from reads; skipping {reads.name}", file=sys.stderr)
        return None
    rel = Path("..") if out.parent == run_dir else Path("../..")
    ts, rs = transcripts(tpath), spans(reads)
    meta_file = run_dir / "meta.json"
    desc = json.loads(meta_file.read_text()).get("description", "") if meta_file.exists() else ""
    envs = {m.get("env") for _, _, m in ts.values()}
    run = Run(out, title=f"{run_dir.parent.name}/{run_dir.name}: {reads.name}", description=desc, label_order=["env", "arm", "positive"],
              reads_file=reads.name)  # fmt: skip
    run.reset()
    n = 0
    for i, (off, length, m) in ts.items():
        if i not in rs:
            continue
        sp, pos = rs[i]
        labels = {k: m[k] for k in ("env", "arm", "item", "positive", "thinking") if isinstance(m.get(k), (str, int, float, bool))}
        src = EPISODE.match(i)
        tokens = {
            "ref": hashlib.sha1(i.encode()).hexdigest()[:16], "n": m["n"], "reads": len(pos),
            "src": {"transcript": [str(rel / tpath.name), off, length], "reads": [[str(rel / reads.name), a, b] for a, b in sp]},
        }  # fmt: skip
        group = m.get("arm") if len(envs) == 1 else "/".join(str(m.get(k)) for k in ("env", "arm") if m.get(k))
        run.add(id=i, calls=[], group=group or None, labels=labels, tokens=tokens, extra={"source": src.group(0)} if src else None)
        n += 1
    print(f"{out}: {n} transcripts with reads ({len(ts) - n} without), {len(rs) - n} read ids without a transcript line", file=sys.stderr)
    return out


def import_run(run_dir: Path) -> list[Path]:
    out = []
    for tpath in sorted(run_dir.glob("*.transcripts.jsonl")):
        reads = tpath.with_name(tpath.name.removesuffix(".transcripts.jsonl") + ".jsonl")
        if reads.exists() and (made := import_reads(run_dir, reads, tpath)):
            out.append(made)
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        import_run(Path(d))
