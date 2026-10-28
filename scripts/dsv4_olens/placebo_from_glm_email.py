"""Placebo reads pool for cfbench's `olens_placebo_agentic` on the DSV4 alignment_faking reads: Celeste's GLM-5.3 RL O-lens
dense email reads (scripts/glm_olens_dense/ output) converted to cfbench's reads format (cfbench/olens.py). cfbench answers
each placebo O-lens call from a run of another env in the same pool, so the ids must not be alignment_faking sources. Her
`readouts_dense.parquet` has one row per unique token prefix, `dense_prefix_dups.parquet` maps every other episode's copy of
that position to it, `episodes.parquet` has each episode's `row_index` into the source run's rows.jsonl. Reads are cut to
greedy + 2 samples with no FVE to look like the DSV4 reads; `char_offset` via scripts/celeste_olens_reads.char_offset
(template tokens get field/char_offset null). Requests aren't marked `positions: "all"`, like the DSV4 requests.

    uv run python scripts/dsv4_olens/placebo_from_glm_email.py data/runs/olens_reads/2026-10-25_glm53_email_dense \\
        --slug placebo-glm-email-for-dsv4-af-v7 --per-arm 8
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from celeste_olens_reads import LAYER, TEMPLATE, char_offset
from runs import new_run

SAMPLES = ("rl_greedy", "rl_s1", "rl_s2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="dense reads run dir with one subdir per source run (readouts_dense.parquet etc.)")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--per-arm", type=int, default=8, help="episodes per (source run, arm)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    subdirs = sorted(p for p in args.src.iterdir() if (p / "readouts_dense.parquet").exists())
    run = new_run(
        "olens_reads",
        args.slug,
        description=f"Placebo pool for cfbench's olens_placebo_agentic on the DSV4 alignment_faking reads: Celeste's GLM-5.3 RL O-lens "
        f"dense email reads ({args.src.name}) in cfbench's format, {args.per_arm} episodes per (run, arm), greedy + 2 samples, no FVE",
        models=[
            {"name": "z-ai/glm-5.3", "served_by": "rollouts: openrouter (z-ai fp8, effort medium)"},
            {
                "name": "ceselder/glm-5.3-olens-lens@098597e16d6edb53d748fc786fd579a76f74ce9a",
                "served_by": f"dense email reads ({args.src.name})",
            },
        ],
        params={"source": str(args.src), "per_arm": args.per_arm, "seed": args.seed, "bullets": list(SAMPLES), "layer": LAYER},
    )
    rng = random.Random(args.seed)
    requests, reads = [], []
    for sub in subdirs:
        dense = pd.read_parquet(sub / "readouts_dense.parquet")
        canon = {(r.tid, r.pos): r for r in dense[dense.view != "late"].itertuples(index=False)}
        by_tid = {}  # tid -> [(pos, dense row)]: its own rows plus the canonical row of each duplicated prefix
        for (tid, pos), r in canon.items():
            by_tid.setdefault(tid, []).append((pos, r))
        dups = pd.read_parquet(sub / "dense_prefix_dups.parquet")
        for d in dups[dups.canon_in == "readouts_dense"].itertuples(index=False):
            by_tid.setdefault(d.tid, []).append((d.pos, canon[d.canon_tid, d.canon_pos]))
        episodes = pd.read_parquet(sub / "episodes.parquet")
        env, run_name = episodes.run_dir.iloc[0].removeprefix("runs/").split("/")
        rows = [json.loads(line) for line in Path("data", episodes.run_dir.iloc[0], "rows.jsonl").read_text().splitlines()]
        picked = []
        for _, group in episodes.groupby("arm"):
            tids = sorted(group.tid)
            rng.shuffle(tids)
            picked += tids[: args.per_arm]
        row_index = dict(zip(episodes.tid, episodes.row_index))
        for tid in sorted(picked):
            source, row, n = f"{env}/{run_name}:{row_index[tid]}", rows[row_index[tid]], len(reads)
            for pos, r in sorted(by_tid.get(tid, []), key=lambda x: x[0]):
                bullets = [{"text": getattr(r, s), "fve": None, "coef": None} for s in SAMPLES if getattr(r, s)]
                if r.msg_index < 0 or not bullets:
                    continue
                template = TEMPLATE.fullmatch(r.token) is not None
                reads.append(
                    {
                        "id": source,
                        "token_index": int(pos),
                        "token": r.token,
                        "prefix": r.ctx_left40,
                        "msg_index": int(r.msg_index),
                        "role": r.role,
                        "field": None if template else "content",
                        "char_offset": None if template else char_offset(row["messages"][r.msg_index]["content"], r.ctx_left40),
                        "layer": LAYER,
                        "bullets": bullets,
                        "joint_fve": None,
                        "kind": r.kind,
                    }
                )
            if len(reads) > n:
                requests.append({"id": source, "messages": row["messages"], "arm": row["arm"], "tid": tid})
    (run / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    (run / "reads.jsonl").write_text("".join(json.dumps(x) + "\n" for x in reads))
    print(f"{run}: {len(reads)} reads over {len(requests)} transcripts")


if __name__ == "__main__":
    main()
