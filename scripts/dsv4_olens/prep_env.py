"""requests.jsonl for read.py from environments/ alignment_faking runs (data/envs/alignment_faking/<run>): one episode per
(arm, item) across the runs given (a top-up pools with its base run), its lowest sample (`--samples all` keeps every sample,
with `:s<sample>` on the id; `--arms public` keeps only those arms), items --items. Each request keeps
`run` and `episode`, so a read maps back to the env_runs loader's `alignment_faking/<run>:<episode>`. The reply is cut to
its first --reply-chars characters: read positions before the cut see the same activations (causal), and long complying
replies would otherwise dominate the read count.

    python scripts/dsv4_olens/prep_env.py --envs data/envs/alignment_faking --runs 2026-10-27_v7-dsv4flash-nocot ... \
        --items 0:100 --out data/runs/olens_reads/<run>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run by path: the repo root isn't on sys.path
from environments._lib.runfolder import active_labels

ap = argparse.ArgumentParser()
ap.add_argument("--envs", required=True)
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--items", default="0:50")
ap.add_argument("--reply-chars", type=int, default=120)
ap.add_argument("--arms", default="", help="comma-separated arms to keep (default: all)")
ap.add_argument("--samples", default="lowest", choices=["lowest", "all"], help="one episode per (arm, item), or every sample")
a = ap.parse_args()

lo, hi = map(int, a.items.split(":"))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
reqs: list[dict] = []
for name in a.runs:
    run = Path(a.envs) / name
    lf, lab = active_labels(run)
    for f in sorted(run.glob("episodes/*.jsonl")):
        for line in f.read_text().splitlines():
            e = json.loads(line)
            q = int(e["item"][1:])
            if not lo <= q < hi:
                continue
            version, _, stage = e["arm"].rpartition("/")
            if a.arms and stage not in a.arms.split(","):
                continue
            msgs = [{"role": m["role"], "content": m["content"]} for m in e["messages"]]
            reply = msgs[-1]["content"]
            msgs[-1]["content"] = reply[: a.reply_chars]
            v = lab[e["id"]]
            reqs.append(
                {
                    "id": f"{version or 'original'}:{stage}:{e['item']}" + (f":s{e['sample']}" if a.samples == "all" else ""),
                    "run": name,
                    "episode": e["id"],
                    "version": version or "original",
                    "arm": stage,
                    "item": e["item"],
                    "sample": e["sample"],
                    "positive": bool(v["positive"]),
                    "verdict": v["verdict"],
                    "reply_chars": len(reply),
                    "label_file": lf,
                    "messages": msgs,
                }
            )
if a.samples == "lowest":  # one episode per (arm, item) across the pooled runs: the lowest sample, ties to the run listed first
    first = {}
    order = {name: i for i, name in enumerate(a.runs)}
    for r in reqs:
        k = (r["arm"], r["item"])
        if k not in first or (r["sample"], order[r["run"]]) < (first[k]["sample"], order[first[k]["run"]]):
            first[k] = r
    reqs = list(first.values())
reqs = sorted(reqs, key=lambda r: (r["version"], r["arm"], int(r["item"][1:]), r["sample"]))
(out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
print(len(reqs), "requests;", sum(r["positive"] for r in reqs), "positive ->", out / "requests.jsonl")
