"""Requests for scripts/olens_read_requests.py on the harness_grid Qwen3.6-27B runs (Kylie 2026-10-30, the Qwen side of the GLM
grader-reads analysis): every arm x item prompt at its "delims" positions (priority 0), then whole thinking-off episodes in a
uniform random order (priority 1), so any prefix of the episode rows is a random sample. At temperature 1 the reader keys its
cache by transcript, so shared prefixes would be re-read per row; this script dedupes them itself: each row lists explicit
`positions`, only those whose token prefix no earlier row reads (prefix_keys as in the reader). Episode rows list only the
positions past the prompt (msg_index >= 2); their prompt is read by the prompt rows.

    uv run python scripts/qwen_olens/prep_harness_grid.py --out REQ.jsonl [--runs 'data/envs/harness_grid/*_qwen36-full-thinkoff']
"""

import argparse
import glob
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[2] / "src")]
from counterfactual_investigator_agent_with_olens.render import render_request
from environments._lib.index import latest_labels
from environments._lib.runfolder import read_episodes

PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")  # as scripts/olens_read_requests.py
DELIM = re.compile(r"^\s*[^\w\s]+\s*$|^\s*\n\s*$")


def delims(tok, ids):
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    special = set(tok.all_special_ids) | set(tok.added_tokens_decoder)
    toks = [tok.decode([i]) for i in ids]
    return [i for i, x in enumerate(ids) if x == im_end or x in special or PUNCT.match(toks[i]) or DELIM.match(toks[i])]


def keys(ids, positions):
    h, out, want = hashlib.blake2b(digest_size=16), {}, set(positions)
    for i, x in enumerate(ids):
        h.update(int(x).to_bytes(4, "little"))
        if i in want:
            out[i] = h.copy().digest()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default="data/envs/harness_grid/*_qwen36-full-thinkoff")
    ap.add_argument("--seed", type=int, default=20261030)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    eps = []
    for d in sorted(glob.glob(a.runs)):
        _, m = latest_labels(Path(d))
        for e in read_episodes(Path(d)):
            if "positive" in (m.get(e["id"]) or {}):
                eps.append((Path(d).name, e))
    seen, rows, n_prompt = set(), [], 0
    prompts = {}  # (arm, item) -> one episode's system + user turns
    for run, e in eps:
        prompts.setdefault((e["arm"], e["item"]), (run, e))
    for (arm, item), (run, e) in sorted(prompts.items()):
        row = {"messages": e["messages"][:2], "tools": e["tools"]}
        _, ids, _ = render_request(tok, row)
        ks = keys(ids, delims(tok, ids))
        new = [p for p, k in ks.items() if k not in seen]
        seen.update(ks.values())
        if new:
            rows.append({"id": f"prompt/{arm}/{item}", "env": "harness_grid", "arm": f"p{len(rows):05d}", "priority": 0, **row,
                         "positions": new, "source": f"envs/harness_grid/{run}/{e['id']}"})  # fmt: skip
            n_prompt += len(new)
    random.Random(a.seed).shuffle(eps)
    n_ep = 0
    for i, (run, e) in enumerate(eps):
        row = {"messages": e["messages"], "tools": e["tools"]}
        _, ids, where = render_request(tok, row)
        ks = keys(ids, [p for p in delims(tok, ids) if where[p][0] >= 2])
        new = [p for p, k in ks.items() if k not in seen]
        seen.update(ks.values())
        rows.append({"id": f"envs/harness_grid/{run}/{e['id']}", "env": "harness_grid", "arm": f"e{i:05d}", "priority": 1, **row,
                     "positions": new})  # fmt: skip
        n_ep += len(new)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    n_rows = sum(r["priority"] == 0 for r in rows)
    print(f"{n_prompt:,} prompt positions in {n_rows} rows; {n_ep:,} episode positions in {len(eps)} episodes "
          f"(mean {n_ep / max(1, len(eps)):.0f})")  # fmt: skip


if __name__ == "__main__":
    main()
