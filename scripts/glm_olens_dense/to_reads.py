"""A dense GLM-5.3 O-lens upload (collect.py's tables, one subdir per source run) as batch reads: RUN/reads.jsonl +
RUN/reads.transcripts.jsonl (docs/olens-reads-format.md), so `expview/importers/reads.py RUN` makes it viewable.

    uv run --with transformers --with jinja2 --with pandas --with pyarrow python scripts/glm_olens_dense/to_reads.py \
        data/runs/olens_reads/<upload> [...]
    python3 expview/importers/reads.py data/runs/olens_reads/<upload>

One transcript per episode view in each subdir's episodes.parquet: tokens are `rendered_text` re-tokenized with zai-org/GLM-5.3
(cut at 32,768 like prep.py), ids `<env>/<run>:<line>` as in scripts/celeste_olens_reads.py, `|late` for Celeste's approximate
tail view. Read rows: the view's own readouts_dense rows plus its dense_prefix_dups (the canonical read, with `dup_of`), layer 51,
bullets = rl greedy + 4 samples with no FVE. Every row's `token` is checked against tokens[token_index]; mismatches are counted
and left out. Canonical reads of episodes outside the sample (no rendered_text) appear only through the dups that point at them.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

CAP = 32768  # prep.py's context cap
LAYER = 51
BULLETS = ["rl_greedy", "rl_s1", "rl_s2", "rl_s3", "rl_s4"]
BLOCK = {"<|system|>": "system", "<|user|>": "user", "<|assistant|>": "assistant", "<|observation|>": "tool"}
INNER = {"<think>": ("</think>", "reasoning"), "<tool_call>": ("</tool_call>", "assistant tool call")}  # spans of an assistant block


def sections(toks: list[str]) -> list[list]:
    """[name, start, end] per GLM template block (a marker token opens it), an assistant block split into reasoning / tool call /
    reply. A marker followed by <think> or <tool_call> goes with that span. [gMASK]<sop> joins the first system block."""
    out, block, inner = [], "system", None
    for i, t in enumerate(toks):
        new = t in BLOCK
        if new:
            block, inner = BLOCK[t], None
            if block == "assistant" and i + 1 < len(toks) and toks[i + 1] in INNER:
                inner = INNER[toks[i + 1]]
        elif block == "assistant" and inner is None and t in INNER:
            inner = INNER[t]
        name = inner[1] if inner else block
        if (new and i > 2) or not out or out[-1][0] != name:
            out.append([name, i, i + 1])
        else:
            out[-1][2] = i + 1
        if inner and t == inner[0]:
            inner = None
    return out


def episode_id(run_dir: str, row_index: int, tid: str) -> str:
    _, env, run = run_dir.split("/")  # runs/<env>/<run>
    return f"{env}/{run}:{row_index}" + ("|late" if tid.endswith(":late") else "")


def labels(labels_json: str) -> dict:
    """The source row's short scalar fields (prep.py's labels_json is the rows.jsonl line minus messages)."""
    d = json.loads(labels_json)
    return {k: v for k, v in d.items() if v is None or isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 80)}


def convert_part(part: Path, tok, fr, ft, stats: dict):
    E = pd.read_parquet(part / "episodes.parquet")
    cols = ["tid", "pos", "token", "role", "kind", "run_dir", "row_index", "labels_json", "positive", "positive_source", "config", *BULLETS]
    R = pd.read_parquet(part / "readouts_dense.parquet", columns=cols)
    R = R.assign(**{c: R[c].astype(object) for c in BULLETS})  # str dtype turns None into NaN
    D = pd.read_parquet(part / "dense_prefix_dups.parquet")
    meta = R.drop_duplicates(["run_dir", "row_index"]).set_index(["run_dir", "row_index"])
    ids = {r.tid: episode_id(r.run_dir, r.row_index, r.tid) for r in R.drop_duplicates("tid").itertuples()}
    reads = {}
    for r in R.itertuples():
        bullets = [{"text": t, "fve": None} for t in (getattr(r, c) for c in BULLETS) if isinstance(t, str) and t]
        reads[(r.tid, int(r.pos))] = {"token": r.token, "layer": LAYER, "bullets": bullets, "joint_fve": None, "role": r.role,
                                      "kind": r.kind}  # fmt: skip
    own = R.groupby("tid").pos.apply(list).to_dict()
    dups = {tid: list(zip(g.pos, g.canon_tid, g.canon_pos)) for tid, g in D.groupby("tid")}
    stats["unsampled_canon_tids"] += len(set(R.tid) - set(E.tid))
    for e in E.itertuples():
        i = episode_id(e.run_dir, e.row_index, e.tid)
        toks = [tok.decode([t]) for t in tok(e.rendered_text, add_special_tokens=False).input_ids[:CAP]]
        if len(toks) != min(e.n_tokens_full, CAP):
            stats["length_mismatch"].append((i, len(toks), e.n_tokens_full))
        rows = [(int(p), e.tid, int(p)) for p in own.get(e.tid, [])] + [(int(p), ct, int(cp)) for p, ct, cp in dups.get(e.tid, [])]
        if sorted(p for p, _, _ in rows) != sorted(int(p) for p in e.positions):
            stats["positions_differ"].append(i)
        n = 0
        for p, ct, cp in sorted(rows):
            x = reads[(ct, cp)]
            if p >= len(toks) or toks[p] != x["token"]:
                stats["token_mismatch"].append((i, p, x["token"], toks[p] if p < len(toks) else None))
                continue
            row = {"id": i, "token_index": p, **x} | ({"dup_of": [ids[ct], cp]} if ct != e.tid else {})
            fr.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            stats["dup_rows" if ct != e.tid else "own_rows"] += 1
        m = meta.loc[(e.run_dir, e.row_index)]
        line = {"id": i, **labels(m.labels_json)}
        line |= {"env": e.run_dir.split("/")[1], "config": m.config, "arm": e.arm, "item": e.item, "positive": bool(m.positive),
                 "positive_source": m.positive_source, "view": "late" if e.tid.endswith(":late") else "prefix",
                 "truncated": bool(e.truncated), "n_tokens_full": int(e.n_tokens_full), "source_run": e.run_dir,
                 "row_index": int(e.row_index), "tid": e.tid, "reads": n, "olens_part": part.name,
                 "tokens": toks, "sections": sections(toks)}  # fmt: skip
        ft.write(json.dumps(line, ensure_ascii=False) + "\n")
        stats["transcripts"] += 1


def main():
    tok = AutoTokenizer.from_pretrained("zai-org/GLM-5.3")
    for run in map(Path, sys.argv[1:]):
        stats = {"transcripts": 0, "own_rows": 0, "dup_rows": 0, "unsampled_canon_tids": 0}
        stats |= {"token_mismatch": [], "length_mismatch": [], "positions_differ": []}
        with open(run / "reads.jsonl", "w") as fr, open(run / "reads.transcripts.jsonl", "w") as ft:
            for part in sorted(p.parent for p in run.glob("*/episodes.parquet")):
                convert_part(part, tok, fr, ft, stats)
                n = stats["own_rows"] + stats["dup_rows"]
                print(f"{part.name}: {stats['transcripts']} transcripts, {n:,} rows so far", file=sys.stderr)
        print(run.name, json.dumps({k: len(v) if isinstance(v, list) else v for k, v in stats.items()}))
        for k in ("token_mismatch", "length_mismatch", "positions_differ"):
            if stats[k]:
                print(f"  {k}: {stats[k][:5]}")


if __name__ == "__main__":
    main()
