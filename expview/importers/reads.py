"""A batch-reads run as an expview dataset, without copying the reads (docs/olens-reads-format.md).

    python expview/importers/reads.py RUN_DIR

For every X.transcripts.jsonl beside its reads X.jsonl, writes RUN_DIR/expview/ (X = reads) or RUN_DIR/expview/X/: one item
per transcript with reads, whose `tokens.src` holds byte ranges of the two files; the page fetches them from HF.
`scripts/runs.py finish` runs this.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from expview import Run

ID = re.compile(rb'^\{"id": ("(?:[^"\\]|\\.)*")')  # readers write `id` first; parsing only it keeps a 1 GB scan fast
EPISODE = re.compile(r"^[^/:|]+/[^/:|]+:\d+")  # ENV/RUN:LINE, the rollout id Docent links resolve through
ENV_EPISODE = re.compile(r"^([^/:|]+/[^/:|]+):(.*~.*)$")  # ENV/RUN:<episode id>, e.g. overeagerness/2026-10-27_x:calendar.base~0~0


def spans(path: Path) -> dict[str, list[list[int]]]:
    """id -> [[offset, length, rows], ...], one per run of consecutive lines (several if shards were concatenated)."""
    out, off, prev = {}, 0, None
    with open(path, "rb") as f:
        for line in f:
            m = ID.match(line)
            assert m, line[:200]
            i = json.loads(m[1])
            if i == prev:
                out[i][-1][1] += len(line)
                out[i][-1][2] += 1
            else:
                out.setdefault(i, []).append([off, len(line), 1])
            prev, off = i, off + len(line)
    return out


def import_reads(run_dir: Path, reads: Path, transcripts: Path):
    name = reads.name.removesuffix(".jsonl")
    out, rel = (run_dir / "expview", "../") if name == "reads" else (run_dir / "expview" / name, "../../")
    if (out / "items.jsonl").exists() and "reads_file" not in json.loads((out / "run.json").read_text()):
        return print(f"{out}: an expview run not made from reads; skipping {reads.name}")
    rs = spans(reads)
    meta = run_dir / "meta.json"
    desc = json.loads(meta.read_text())["description"] if meta.exists() else ""
    run = Run(out, title=f"{run_dir.parent.name}/{run_dir.name}: {reads.name}", description=desc, label_order=["env", "arm", "positive"],
              reads_file=reads.name)  # fmt: skip
    run.reset()
    off = 0
    with open(transcripts, "rb") as f:
        for line in f:
            t = json.loads(line)
            if sp := rs.pop(t["id"], None):  # pop: a reader may repeat a transcript line after a crash
                src = {"transcript": [rel + transcripts.name, off, len(line)], "reads": [[rel + reads.name, a, n] for a, n, _ in sp]}
                ref = hashlib.sha1(t["id"].encode()).hexdigest()[:16]
                tokens = {"ref": ref, "n": len(t["tokens"]), "reads": sum(r for *_, r in sp), "src": src}
                m, e = EPISODE.match(t["id"]), ENV_EPISODE.match(t["id"])
                # an environments/ episode (`ENV/RUN:<episode id>`) names its run, so the env viewer attaches the reads
                extra = {"source": m[0]} if m else {"source": f"envs/{e[1]}", "episode": e[2]} if e else None
                run.add(id=t["id"], calls=[], group="/".join(str(t[k]) for k in ("env", "arm") if k in t) or None,
                        labels={k: t[k] for k in ("env", "arm", "item", "positive", "thinking") if k in t}, tokens=tokens,
                        extra=extra)  # fmt: skip
            off += len(line)
    print(f"{out}: {run_dir.name}/{reads.name}, {len(rs)} read ids without a transcript line")


if __name__ == "__main__":
    for tpath in sorted(Path(sys.argv[1]).glob("*.transcripts.jsonl")):
        reads = tpath.with_name(tpath.name.removesuffix(".transcripts.jsonl") + ".jsonl")
        if reads.exists():
            import_reads(Path(sys.argv[1]), reads, tpath)
