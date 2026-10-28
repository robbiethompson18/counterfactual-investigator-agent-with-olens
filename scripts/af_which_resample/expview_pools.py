"""expview items for the shared-position sample pools: per (model, version) the shared user-turn prefix with every pool
sample at each position, then the 4 samples a few cases got (from an inputs run).

    python scripts/af_which_resample/expview_pools.py INPUTS_RUN POOL_RUN [--cases 4]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "expview"))
from cfbench.evals.af_refusals import eval as af
from expview import Run, read

ap = argparse.ArgumentParser()
ap.add_argument("inputs", type=Path)
ap.add_argument("pool_run", type=Path)
ap.add_argument("--cases", type=int, default=4)
a = ap.parse_args()
pool = [json.loads(line) for p in a.pool_run.glob("pool_*.jsonl") for line in p.read_text().splitlines()]
model = pool[0]["model"]
name = af.MODELS[model]
cases = [json.loads(line) for line in (a.inputs / "cases.jsonl").read_text().splitlines() if model in line]
reads = {}
for line in (a.inputs / "reads/reads.jsonl").read_text().splitlines():
    r = json.loads(line)
    reads.setdefault(r["id"], []).append(r)
run = Run(
    a.pool_run / "expview",
    title=f"{name} O-lens sample pool at af_which's shared user-turn positions",
    description=(
        "Every af_which case of a version shares the user turn up to `<user_input>`, so the model's activations there are "
        "identical and the original reads were one lens draw copied into every case. This pool is many lens samples at each "
        "of those positions (blank ones dropped); each af_which case now gets its own 4 of them. Items `pool/<version>`: "
        "every sample per position. Items `case/<version>/<item>`: the 4 that case got (inputs run "
        f"`{a.inputs.name}`). The AdvBench request that follows is not shown."
    ),
)
for version in ("original", "v7"):
    pos = sorted((x for x in pool if x["version"] == version), key=lambda x: x["idx"])
    toks = [x["token"] for x in pos]
    run.add(
        id=f"pool/{version}",
        group=version,
        calls=[],
        labels={"positions": len(pos), "min samples": min(len(x["samples"]) for x in pos)},
        tokens={
            "tokens": toks,
            "sections": [["user", 0, len(toks)]],
            "reads": {str(i): [read("O-lens pool", x["samples"], label=f"{len(x['samples'])} samples")] for i, x in enumerate(pos)},
        },
    )
    for c in sorted((c for c in cases if c["version"] == version), key=lambda c: int(c["item"][1:]))[: a.cases]:
        start = next(af.REQUEST.finditer(c["messages"][1]["content"])).start()
        shared = sorted(
            (r for r in reads[c["source"]] if r["char_offset"] is not None and r["char_offset"] < start), key=lambda r: r["char_offset"]
        )
        assert [r["token"] for r in shared] == toks
        run.add(
            id=f"case/{version}/{c['item']}",
            group=version,
            calls=[],
            tokens={
                "tokens": toks,
                "sections": [["user", 0, len(toks)]],
                "reads": {str(i): [read("O-lens", [b["text"] for b in r["bullets"]], label="this case's 4")] for i, r in enumerate(shared)},
            },
        )
print(run.path)
