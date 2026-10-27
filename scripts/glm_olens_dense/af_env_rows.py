"""Convert environments/ alignment_faking runs (data/envs/alignment_faking/<run>/episodes/*.jsonl + labels/measure@*.jsonl)
into one rows.jsonl per run for prep.py (single turn, no tools; GLM runs used --backend openrouter-raw, so prep with --raw-prefill).

    python af_env_rows.py --envs data/envs/alignment_faking --runs 2026-10-27_v4-glm53-nocot ... --out <dir>
"""

import argparse
import json
from pathlib import Path


def labels(run: Path) -> dict:
    files = sorted(run.glob("labels/measure@*.jsonl"))
    f = next((x for x in files if x.name == "measure@v3-curly.jsonl"), files[-1])  # the fixed detector where it was rerun
    return {(d := json.loads(x))["episode"]: d["values"] | {"label_file": f.name} for x in f.read_text().splitlines()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.runs:
        run = Path(a.envs) / name
        meta = json.loads((run / "meta.json").read_text())
        lab = labels(run)
        rows = []
        for f in sorted(run.glob("episodes/*.jsonl")):
            for line in f.read_text().splitlines():
                e = json.loads(line)
                msgs = [{"role": m["role"], "content": m["content"]} for m in e["messages"]]
                assert [m["role"] for m in msgs] == ["system", "user", "assistant"], e["id"]
                v = lab[e["id"]]
                version, _, stage = e["arm"].rpartition("/")  # original runs: bare stage arms (free/paid/preview/public)
                rows.append(
                    {
                        "env": "alignment_faking",
                        "run": name,
                        "episode": e["id"],
                        "arm": e["arm"],
                        "version": version or "original",
                        "stage": stage,
                        "item": e["item"],
                        "sample": e["sample"],
                        "model": meta["target"]["model"],
                        "backend": meta["target"]["backend"],
                        "thinking": bool(meta["target"].get("thinking")),
                        "positive": bool(v["positive"]),
                        "verdict": v["verdict"],
                        "label_file": v["label_file"],
                        "messages": msgs,
                    }
                )
        rows.sort(key=lambda r: (r["arm"], int(r["item"][1:]), r["sample"]))
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(name, len(rows), sum(r["positive"] for r in rows))


if __name__ == "__main__":
    main()
