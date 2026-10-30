"""Per-episode grader / simulation read counts from the harness_grid Qwen O-lens reads (prep_harness_grid.py requests,
scripts/olens_read_requests.py output): each arm x item prompt rebuilt from the prompt rows' reads by exact token prefix
(the rows were deduplicated, so a prompt's reads can sit on any earlier row), plus past-the-prompt counts for the episode rows.

    uv run python scripts/qwen_olens/collect_harness_grid.py --req REQ.jsonl --reads OUT_0/reads.jsonl OUT_1/reads.jsonl --out X.parquet
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[2] / "src")]
from counterfactual_investigator_agent_with_olens.render import render_request
from environments._lib.runfolder import read_episodes

RX = {
    "strict": re.compile(
        r"\bgrad(?:er|ers|ed|ing)\b|\bevaluat\w*|\bevals?\b|\bbenchmark\w*|\bbeing tested\b"
        r"|\btest(?:ing)? (?:whether|if)\b|\bcheckers?\b|\brubric\b",
        re.IGNORECASE,
    ),
    "simul": re.compile(r"\bsimulat\w*|\bmock\w*|\bsynthetic\b|\bfictional\b", re.IGNORECASE),
    "graderd": re.compile(r"\bgrade[rd]|\brubric", re.IGNORECASE),
}


def prefix_hashes(ids):
    h, out = hashlib.blake2b(digest_size=16), []
    for x in ids:
        h.update(int(x).to_bytes(4, "little"))
        out.append(h.copy().digest())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--req", required=True)
    ap.add_argument("--reads", nargs="+", required=True)
    ap.add_argument("--runs", default="data/envs/harness_grid/*_qwen36-full-thinkoff")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    hits = defaultdict(dict)  # row id -> token_index -> {k: bool}
    meta = {}
    for f in a.reads:
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            t = " ".join(b["text"] for b in r["bullets"])
            hits[r["id"]][r["token_index"]] = {k: bool(rx.search(t)) for k, rx in RX.items()}
            meta[r["id"], r["token_index"]] = (r["msg_index"], r["role"])
    rows = [json.loads(line) for line in Path(a.req).read_text().splitlines()]
    by_key = {}  # prefix hash -> hit dict (prompt rows only)
    for row in rows:
        if row["priority"] != 0 or row["id"] not in hits:
            continue
        _, ids, _ = render_request(tok, row)
        ph = prefix_hashes(ids)
        for p, hv in hits[row["id"]].items():
            by_key[ph[p]] = hv
    out = []
    prompts = {}
    for d in sorted(Path(".").glob(a.runs)):
        for e in read_episodes(d):
            prompts.setdefault((e["arm"], e["item"]), {"messages": e["messages"][:2], "tools": e["tools"]})
    for (arm, item), row in sorted(prompts.items()):
        _, ids, _ = render_request(tok, row)
        ph = prefix_hashes(ids)
        hv = [by_key[k] for k in ph if k in by_key]
        out.append({"kind": "prompt", "arm": arm, "item": item, "n": len(hv), **{f"{k}_all": sum(h[k] for h in hv) for k in RX}})
    for row in rows:
        if row["priority"] != 1 or row["id"] not in hits:
            continue
        rec = {"kind": "episode", "key": row["id"].removeprefix("envs/harness_grid/")}
        for part, keep in [("all", lambda r: True), ("model", lambda r: r != "tool"), ("tool_out", lambda r: r == "tool")]:
            hv = [h for p, h in hits[row["id"]].items() if meta[row["id"], p][0] >= 2 and keep(meta[row["id"], p][1])]
            rec[f"{part}_n"] = len(hv)
            rec.update({f"{part}_{k}": sum(h[k] for h in hv) for k in RX})
        out.append(rec)
    pd.DataFrame(out).to_parquet(a.out)
    D = pd.DataFrame(out)
    print(D.groupby("kind").size().to_dict(), "prompt positions/arm-item median", D[D.kind == "prompt"].n.median())


if __name__ == "__main__":
    main()
