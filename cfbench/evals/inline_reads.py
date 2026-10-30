"""Every O-lens and NLA read of the 2026-10-29 inline evals' prompts, at the default read positions (docs/read-positions.md).

The 2026-10-29 runs read hand-picked positions; the 2026-10-30 fill run (scripts/fill_read_requests.py) read the rest of
the default set under the same ids: every PUNCT token, each MCQ option's newline and the ten tail tokens of the
thinking-off render, layers 32/40/48/52/56/60, O-lens and NLA. `rows(name, reader)` is the union, one row per
(id, token_index, layer), so a condition that wants every position needs no splicing:

    from cfbench.evals import inline_reads
    rows = inline_reads.rows("supergpqa", "olens")   # or "nla"; names are the keys of SETS

The existing conditions of these evals still read only their old runs (their blocks are laid out for the old positions).
"""

import json
from functools import cache
from pathlib import Path

RUNS = Path("data/runs")
FILL = "2026-10-30_inline-fill"
# set -> (O-lens runs, NLA runs) of 2026-10-29; the O-lens runs' requests are the set's prompts
SETS = {
    "cutoff": (["2026-10-29_cutoff200-tail", "2026-10-29_cutoff800-tail"], ["2026-10-29_cutoff200-tail", "2026-10-29_cutoff800-tail"]),
    "favorites": (["2026-10-29_favorites-v3-tail"], ["2026-10-29_favorites-v3-tail"]),
    "mcq_unselected": (["2026-10-29_mcq-unselected-nohint", "2026-10-29_mcq-unselected-nohint-b2"], ["2026-10-29_mcq-unselected-nohint"]),
    "supergpqa": (["2026-10-29_supergpqa-3k"], ["2026-10-29_supergpqa-3k"]),
}


def ids(name: str) -> set[str]:
    return {json.loads(line)["id"] for run in SETS[name][0] for line in (RUNS / "olens_reads" / run / "requests.jsonl").open()}


def paths(name: str, reader: str) -> list[Path]:
    """The reads files holding `name`'s reads: its old runs, then the fill run."""
    old = SETS[name][0 if reader == "olens" else 1]
    return [RUNS / f"{reader}_reads" / run / "reads.jsonl" for run in [*old, FILL]]


@cache
def rows(name: str, reader: str) -> list[dict]:
    """Every read of the set's prompts, old and fill, sorted by (id, token_index, layer); asserts no read is there twice."""
    keep, out, seen = ids(name), [], set()
    for path in paths(name, reader):
        for line in path.open():
            r = json.loads(line)
            if r["id"] in keep:
                key = (r["id"], r["token_index"], r["layer"])
                assert key not in seen, (path, key)
                seen.add(key)
                out.append(r)
    return sorted(out, key=lambda r: (r["id"], r["token_index"], r["layer"]))
