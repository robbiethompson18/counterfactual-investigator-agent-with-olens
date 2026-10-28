"""requests.jsonl for read.py from environments/ alignment_faking runs (data/envs/alignment_faking/<run>): one episode per
(arm, item) across the runs given (a top-up pools with its base run), its lowest sample, items --items. Each request keeps
`run` and `episode`, so a read maps back to the env_runs loader's `alignment_faking/<run>:<episode>`. The reply is cut to
its first --reply-chars characters: read positions before the cut see the same activations (causal), and long complying
replies would otherwise dominate the read count.

    python scripts/dsv4_olens/prep_env.py --envs data/envs/alignment_faking --runs 2026-10-27_v7-dsv4flash-nocot ... \
        --items 0:100 --out data/runs/olens_reads/<run>
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--envs", required=True)
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--items", default="0:50")
ap.add_argument("--reply-chars", type=int, default=120)
a = ap.parse_args()

lo, hi = map(int, a.items.split(":"))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
reqs = []
for name in a.runs:
    run = Path(a.envs) / name
    files = sorted(run.glob("labels/measure@*.jsonl"))
    lf = next((x for x in files if x.name == "measure@v3-curly.jsonl"), files[-1])
    lab = {(d := json.loads(x))["episode"]: d["values"] for x in lf.read_text().splitlines()}
    for f in sorted(run.glob("episodes/*.jsonl")):
        for line in f.read_text().splitlines():
            e = json.loads(line)
            q = int(e["item"][1:])
            if not lo <= q < hi:
                continue
            version, _, stage = e["arm"].rpartition("/")
            msgs = [{"role": m["role"], "content": m["content"]} for m in e["messages"]]
            reply = msgs[-1]["content"]
            msgs[-1]["content"] = reply[: a.reply_chars]
            v = lab[e["id"]]
            reqs.append(
                {
                    "id": f"{version or 'original'}:{stage}:{e['item']}",
                    "run": name,
                    "episode": e["id"],
                    "version": version or "original",
                    "arm": stage,
                    "item": e["item"],
                    "sample": e["sample"],
                    "positive": bool(v["positive"]),
                    "verdict": v["verdict"],
                    "reply_chars": len(reply),
                    "label_file": lf.name,
                    "messages": msgs,
                }
            )
first = {}  # one episode per (arm, item) across the pooled runs: the lowest sample, ties to the run listed first
order = {name: i for i, name in enumerate(a.runs)}
for r in reqs:
    k = (r["arm"], r["item"])
    if k not in first or (r["sample"], order[r["run"]]) < (first[k]["sample"], order[first[k]["run"]]):
        first[k] = r
reqs = sorted(first.values(), key=lambda r: (r["version"], r["arm"], int(r["item"][1:])))
(out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
print(len(reqs), "requests;", sum(r["positive"] for r in reqs), "positive ->", out / "requests.jsonl")
