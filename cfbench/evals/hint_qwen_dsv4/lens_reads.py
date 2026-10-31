"""NLA and J++ reads at exactly the O-lens reads' tokens for the four hinted-MCQ tail evals (Kylie, 2026-10-31).

The O-lens blocks of hint_qwen_dsv4{,_before}{,_letter}_tail_inline come from four O-lens read runs (SOURCES): the option
newlines of the hint-after pool (172 + 804 shown runs), of the hint-first pool (952), and the six end-of-prompt tokens of
both (`tail`; its 400 version-less wrong-hint rows are not in any eval and are left out). `requests` copies their
requests.jsonl rows (same render, same explicit token indices) into one file, ids prefixed with the source tag, for one
pass of each reader; `split` cuts a reader's output back into one dir per source (original ids, requests.jsonl beside
reads.jsonl, so inline.py loads it like the O-lens dirs; J++ rows exploded to one row per layer with `top`) and asserts
that every source's (id, token_index, token) set equals the O-lens run's.

    uv run python -m cfbench.evals.hint_qwen_dsv4.lens_reads requests OUT.jsonl
    uv run python -m cfbench.evals.hint_qwen_dsv4.lens_reads split READER_OUT.jsonl RUN_DIR

Readers: J++ (olens.jpp, layers 32..60 step 4, top 10; Kylie: "every 4th layer") and
ceselder's layer-42 NLA (ceselder/qwen3.6-27b-nla-rl, Kylie's pick), at the same tokens; the critic's FVE per read.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from cfbench.evals.hint_qwen_dsv4 import eval as K

SOURCES = {
    "nl": K.READS_NL,
    "nl12k": K.READS_NL_12K,
    "before": K.READS_NL_BEFORE,
    "tail": K.REPO_ROOT / "data/runs/olens_reads/2026-10-29_hint-qwen-dsv4-tail",
}
SEP = "|"
# the split reader runs: one dir per SOURCES tag inside each
RUNS = {
    "nla": K.REPO_ROOT / "data/runs/nla_reads/2026-10-31_hint-qwen-dsv4-ceselder-l42",
    "jlens": K.REPO_ROOT / "data/runs/jpp_reads/2026-10-31_hint-qwen-dsv4-same-tokens",
}


def lens_dir(reader: str, olens_dir: Path) -> Path:
    """The `reader` ("olens", "nla", "jlens") reads at the same tokens as the O-lens run `olens_dir`."""
    if reader == "olens":
        return olens_dir
    return RUNS[reader] / next(t for t, d in SOURCES.items() if d == olens_dir)


def _rows(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def _keep(tag: str, row: dict) -> bool:
    return tag != "tail" or row.get("version") is not None


def requests(out: Path):
    n = 0
    with out.open("w") as f:
        for tag, d in SOURCES.items():
            for r in _rows(d / "requests.jsonl"):
                if _keep(tag, r):
                    assert isinstance(r["positions"], list), (tag, r["id"])
                    f.write(json.dumps(r | {"id": f"{tag}{SEP}{r['id']}"}, ensure_ascii=False) + "\n")
                    n += 1
    print(f"wrote {n} requests to {out}")


def _sites(rows) -> set[tuple[str, int, str]]:
    return {(r["id"], r["token_index"], r["token"]) for r in rows}


def split(reads: Path, run: Path):
    by = defaultdict(list)
    for r in _rows(reads):
        tag, rid = r["id"].split(SEP, 1)
        r["id"] = rid
        if "top" in r and "layers" in r:  # J++: one row per layer, as the O-lens / NLA rows
            by[tag] += [
                {k: v for k, v in r.items() if k not in ("layers", "top")} | {"layer": L, "top": t} for L, t in zip(r["layers"], r["top"])
            ]
        else:
            by[tag].append(r)
    for tag, d in SOURCES.items():
        ids = {q["id"] for q in _rows(d / "requests.jsonl") if _keep(tag, q)}
        want = _sites(r for r in _rows(d / "reads.jsonl") if r["id"] in ids)
        got = _sites(by[tag])
        assert got == want, f"{tag}: {len(got - want)} extra, {len(want - got)} missing sites vs {d.name}"
        (run / tag).mkdir(parents=True, exist_ok=True)
        with (run / tag / "reads.jsonl").open("w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in by[tag])
        with (run / tag / "requests.jsonl").open("w") as f:
            f.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in _rows(d / "requests.jsonl") if _keep(tag, q))
        print(f"{tag}: {len(by[tag])} rows at {len(got)} sites, same as {d.name}")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    requests(Path(a[0])) if cmd == "requests" else split(Path(a[0]), Path(a[1]))
