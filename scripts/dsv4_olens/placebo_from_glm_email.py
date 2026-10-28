"""A placebo reads run for cfbench's `olens_placebo_agentic` on the DSV4 alignment_faking reads, from Celeste's GLM-5.3 RL
O-lens dense email reads (scripts/glm_olens_dense/ output, e.g. data/runs/olens_reads/2026-10-25_glm53_email_dense/).

cfbench's placebo answers every O-lens call from a run of another env in the same reads pool, so the pool needs reads in
its own format (cfbench/olens.py) over transcripts whose `id` isn't an alignment_faking source. Her dense tables hold one
row per unique token prefix (`readouts_dense.parquet`), every other episode's copy of that position mapped through
`dense_prefix_dups.parquet`; `episodes.parquet` has each episode's `row_index` into the source run's rows.jsonl, which has
the messages. Reads are cut to the greedy readout + the first 2 samples with no FVE, to look like the DSV4 reads
(greedy + 2 samples, no reconstructor score). `char_offset` comes from matching the tail of her `ctx_left40` in the
message (scripts/celeste_olens_reads.char_offset); template tokens get field/char_offset null. Requests are not marked
`positions: "all"`: the DSV4 requests aren't either, so under cfbench's default sparse filter both files serve every row
the job read (her dense head reads here, every user-turn and reply token for the DSV4 reads).

    uv run python scripts/dsv4_olens/placebo_from_glm_email.py data/runs/olens_reads/2026-10-25_glm53_email_dense \\
        --slug placebo-glm-email-for-dsv4-af-v7 --per-arm 8
"""

import argparse
import json
import random
import sys
from collections import Counter
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
        f"dense email reads ({args.src.name}: {', '.join(p.name for p in subdirs)}) converted to cfbench's reads format, "
        f"{args.per_arm} episodes per (run, arm), each read cut to greedy + 2 samples with no FVE, layer {LAYER}",
        models=[
            {"name": "z-ai/glm-5.3", "served_by": "rollouts: openrouter (z-ai fp8, effort medium)"},
            {
                "name": "ceselder/glm-5.3-olens-lens@098597e16d6edb53d748fc786fd579a76f74ce9a",
                "served_by": f"Celeste's GLM 5.3 RL O-lens dense email reads ({args.src.name}), cut to 3 bullets, fve null, "
                "as placebo for the DSV4 alignment_faking reads",
            },
        ],
        params={"source": str(args.src), "per_arm": args.per_arm, "seed": args.seed, "bullets": list(SAMPLES), "layer": LAYER},
    )
    rng = random.Random(args.seed)
    requests, reads, stats = [], [], Counter()
    for sub in subdirs:
        dense = pd.read_parquet(sub / "readouts_dense.parquet")
        dense = dense[(dense.view != "late")]
        canon = {(r.tid, r.pos): r for r in dense.itertuples(index=False)}
        dups = pd.read_parquet(sub / "dense_prefix_dups.parquet")
        stats.update(f"canon_in={c}" for c in dups.canon_in)
        dups = dups[dups.canon_in == "readouts_dense"]
        dup_map: dict[str, list[tuple[int, str, int]]] = {}  # tid -> [(pos, canon_tid, canon_pos)]
        for r in dups.itertuples(index=False):
            dup_map.setdefault(r.tid, []).append((r.pos, r.canon_tid, r.canon_pos))
        episodes = pd.read_parquet(sub / "episodes.parquet")
        run_dir = episodes.run_dir.iloc[0]
        assert (episodes.run_dir == run_dir).all(), sub
        env, run_name = run_dir.removeprefix("runs/").split("/")
        rows = [json.loads(line) for line in Path("data", run_dir, "rows.jsonl").read_text().splitlines()]
        picked = []
        for _, group in episodes.groupby("arm"):
            tids = sorted(group.tid)
            rng.shuffle(tids)
            picked += tids[: args.per_arm]
        row_index = dict(zip(episodes.tid, episodes.row_index))
        for tid in sorted(picked):
            source = f"{env}/{run_name}:{row_index[tid]}"
            messages = rows[row_index[tid]]["messages"]
            positions = [(pos, tid, pos) for (t, pos) in canon if t == tid] + dup_map.get(tid, [])
            n = 0
            for pos, ctid, cpos in sorted(positions):
                r = canon[ctid, cpos]
                if r.msg_index < 0:
                    stats["dropped msg_index<0"] += 1
                    continue
                bullets = [{"text": getattr(r, s), "fve": None, "coef": None} for s in SAMPLES if getattr(r, s)]
                if not bullets:
                    stats["dropped no bullets"] += 1
                    continue
                message = messages[r.msg_index]
                assert message["role"] == r.role, (source, pos, message["role"], r.role)
                template = TEMPLATE.fullmatch(r.token) is not None
                offset = None if template else char_offset(message["content"], r.ctx_left40)
                stats["template" if template else ("matched" if offset is not None else "unmatched")] += 1
                reads.append(
                    {
                        "id": source,
                        "token_index": int(pos),
                        "token": r.token,
                        "prefix": r.ctx_left40,
                        "msg_index": int(r.msg_index),
                        "role": r.role,
                        "field": None if template else "content",
                        "char_offset": offset,
                        "layer": LAYER,
                        "bullets": bullets,
                        "joint_fve": None,
                        "kind": r.kind,
                    }
                )
                n += 1
            if n:
                requests.append({"id": source, "messages": messages, "arm": rows[row_index[tid]]["arm"], "tid": tid})
            else:
                stats["episodes without reads"] += 1
    (run / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    (run / "reads.jsonl").write_text("".join(json.dumps(x) + "\n" for x in reads))
    print(f"{run}: {len(reads)} reads over {len(requests)} transcripts; {dict(stats)}")


if __name__ == "__main__":
    main()
