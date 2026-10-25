"""Convert Celeste's GLM-5.3 O-lens readouts (HF ceselder/glm-5.3-olens-counterfactual-investigator, config `evalcue`) into our
reads format (cfbench/olens.py), so cfbench's olens / olens_placebo conditions can use them on our GLM 5.3 evalcue runs.

    uv run python scripts/celeste_olens_reads.py READOUTS.parquet --envs scheming_oversight_blk,dumb_user_exploit_defect --slug ...

Her rows join to ours by run_dir + row_index (0-based line of rows.jsonl), which is cfbench's rollout id `<env>/<run>:<line>`.
Differences from our own reads, all kept visible downstream:
- one layer (51), one lens per output (`--lens rl` or `sft`); a read's bullets are its greedy readout then its 4 samples, with
  no FVE (`fve: None`): her lens has no reconstructor score.
- ~12 positions per episode (final user turn, arm-diff span edges, final reasoning/answer, pre-action, fill), not every stop.
- her `pos` indexes her GLM-template render, so `char_offset` is found by matching the end of `ctx_left40` inside the
  message field (None if it doesn't match uniquely: cfbench then places the read at the message's end).
- `view == "late"` rows (approximate views of >32k-token episodes) are dropped.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from runs import finish, new_run

LAYER = 51  # her lens reads the output of GLM-5.3 block 51
FIELD = {"reasoning": "reasoning_content", "tool_call": "tool_calls"}  # her role -> our message field (else "content")
TEMPLATE = re.compile(r"<\|[^|>]*\|>|</?think>|\[gMASK\]|<sop>")
TAILS = (80, 50, 30, 18, 10)  # suffix lengths of ctx_left40 to try, longest first; the first unique match wins


def char_offset(text: str, ctx_left: str) -> int | None:
    """Index in `text` of the read token's last character, from the tail of the read's left context."""
    tail_src = TEMPLATE.split(ctx_left)[-1]  # only the part inside this message
    for k in TAILS:
        tail = tail_src[-k:]
        if len(tail) < min(k, 8):
            continue
        hits = [m.start() for m in re.finditer(re.escape(tail), text)]
        if len(hits) == 1:
            return hits[0] + len(tail) - 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("readouts", type=Path, help="her evalcue/readouts.parquet")
    ap.add_argument("--envs", required=True, help="comma list of env names (her `env` column)")
    ap.add_argument("--lens", choices=["rl", "sft"], default="rl")
    ap.add_argument("--slug", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.readouts)
    df = df[df.env.isin(args.envs.split(",")) & (df.view != "late") & (df.msg_index >= 0)]
    run = new_run(
        "olens_reads",
        args.slug,
        description=f"Celeste's GLM-5.3 {args.lens.upper()} O-lens readouts (layer {LAYER}, greedy + 4 samples, no FVE) converted to "
        f"cfbench's reads format for {args.envs}",
        models=[
            {"name": "z-ai/glm-5.3", "served_by": "rollouts: openrouter (z-ai fp8, effort medium)"},
            {"name": "ceselder/glm5.3-olens", "served_by": "ceselder/glm-5.3-olens-counterfactual-investigator (HF), config evalcue"},
        ],
        params={"lens": args.lens, "layer": LAYER, "envs": args.envs, "source": str(args.readouts)},
    )
    rows_cache: dict[str, list[dict]] = {}
    requests, reads, stats = {}, [], Counter()
    for r in df.itertuples():
        env, run_name = r.run_dir.removeprefix("runs/").split("/")
        if r.run_dir not in rows_cache:
            rows_cache[r.run_dir] = [json.loads(line) for line in Path("data", r.run_dir, "rows.jsonl").read_text().splitlines()]
        row = rows_cache[r.run_dir][r.row_index]
        assert (row["arm"], str(row["item"])) == (r.arm, str(r.item)), f"{r.run_dir}:{r.row_index} doesn't match her row"
        source = f"{env}/{run_name}:{r.row_index}"
        requests[source] = row["messages"]
        field = FIELD.get(r.role, "content")
        message = row["messages"][r.msg_index]
        offset = char_offset(message.get(field) or "", r.ctx_left40) if isinstance(message.get(field), str) else None
        stats["matched" if offset is not None else "unmatched"] += 1
        texts = [getattr(r, f"{args.lens}_{k}") for k in ("greedy", "s1", "s2", "s3", "s4")]
        reads.append(
            {
                "id": source,
                "token_index": int(r.pos),
                "token": r.token,
                "prefix": r.ctx_left40,
                "msg_index": int(r.msg_index),
                "role": message["role"],
                "field": field,
                "char_offset": offset,
                "layer": LAYER,
                "bullets": [{"text": t, "fve": None, "coef": None} for t in texts if t],
                "joint_fve": None,
                "kind": r.kind,
            }
        )
    (run / "requests.jsonl").write_text("".join(json.dumps({"id": s, "messages": m}) + "\n" for s, m in requests.items()))
    (run / "reads.jsonl").write_text("".join(json.dumps(x) + "\n" for x in reads))
    print(f"{len(reads)} reads over {len(requests)} episodes; char_offset {dict(stats)}")
    finish(run)


if __name__ == "__main__":
    main()
