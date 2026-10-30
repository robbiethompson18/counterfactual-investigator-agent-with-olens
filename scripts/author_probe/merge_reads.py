"""Merge a reader's output dir (reads.jsonl, reads.transcripts.jsonl, requests.jsonl, reads.provenance.json) into a batch
reads run, so one run holds every read of a dataset (docs/olens-reads-format.md): rows deduplicated by (id, token, layer)
and grouped by transcript, transcripts and requests merged with their positions unioned.

    uv run python -m scripts.author_probe.merge_reads RUN NEW_DIR [NEW_DIR ...]   # RUN may be empty (a fresh runs.py run)
"""

import json
import sys
from pathlib import Path


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def merge(run: Path, news: list[Path]):
    key = lambda r: (r["id"], r["token_index"], r["layer"])
    rows = {key(r): r for d in [run, *news] for r in load(d / "reads.jsonl")}
    tr: dict[str, dict] = {}
    req: dict[str, dict] = {}
    for d in [run, *news]:
        for name, into in (("reads.transcripts.jsonl", tr), ("requests.jsonl", req)):
            for r in load(d / name):
                if r["id"] in into:
                    into[r["id"]]["positions"] = sorted(set(into[r["id"]]["positions"]) | set(r["positions"]))
                else:
                    into[r["id"]] = r
    if not (run / "reads.provenance.json").exists():
        (run / "reads.provenance.json").write_text((news[0] / "reads.provenance.json").read_text())
    (run / "reads.jsonl").write_text("".join(json.dumps(rows[k], ensure_ascii=False) + "\n" for k in sorted(rows)))
    (run / "reads.transcripts.jsonl").write_text("".join(json.dumps(tr[i], ensure_ascii=False) + "\n" for i in sorted(tr)))
    (run / "requests.jsonl").write_text("".join(json.dumps(req[i]) + "\n" for i in sorted(req)))
    n = {}
    for k in rows:
        n[k[0]] = n.get(k[0], 0) + 1
    layers = len({k[2] for k in rows})
    bad = [i for i, r in req.items() if n.get(i, 0) != layers * len(r["positions"])]
    print(f"{len(rows)} reads, {len(tr)} transcripts; incomplete: {len(bad)} {bad[:5]}")


if __name__ == "__main__":
    merge(Path(sys.argv[1]), [Path(x) for x in sys.argv[2:]])
