"""Convert read.py / vllm_read.py rows (id, pos, char, role, token, ctx_left, greedy, s1, s2, layer) into cfbench's reads
format (cfbench/olens.py: id, token_index, token, prefix, msg_index, role, field, char_offset, layer, bullets, joint_fve)
with the `requests.jsonl` of {id, messages} cfbench expects beside it.

The cfbench `id` is the env_runs loader's Rollout source, `alignment_faking/<run>:<episode>`, from the `run` and `episode`
fields prep_env.py kept on each request. `char_offset` is exact: read.py's `char` indexes the rendered string (BOS + system
+ <｜User｜> + user + <｜Assistant｜></think> + reply, read.render), so each message's content starts at a known offset.
Template tokens (<｜User｜>, <｜Assistant｜>, </think>, BOS) get char_offset null, which cfbench places at the message's end.
Bullets are the greedy readout then the samples, with no FVE (Camila's inverter has no reconstructor score).

    python scripts/dsv4_olens/to_cfbench_reads.py RUN_DIR [--reads reads_L30.jsonl] [--out cfbench]
    # RUN_DIR/requests.jsonl (prep_env.py) + RUN_DIR/reads_L30.jsonl -> RUN_DIR/cfbench/{reads,requests}.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

# read.py's render, copied so this runs without torch: BOS + system + <｜User｜> + user + <｜Assistant｜></think> + reply
BOS, USER, ASSISTANT, NO_THINK = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "</think>"
ENV = "alignment_faking"
SAMPLES = ["greedy", "s1", "s2", "s3", "s4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--reads", default="reads_L30.jsonl")
    ap.add_argument("--out", default="cfbench")
    a = ap.parse_args()
    reqs = {r["id"]: r for r in map(json.loads, (a.run_dir / "requests.jsonl").read_text().splitlines())}
    out = a.run_dir / a.out
    out.mkdir(exist_ok=True)
    starts: dict[str, dict[str, tuple[int, int]]] = {}  # request id -> role -> (msg_index, content start in the rendered text)
    with (out / "requests.jsonl").open("w") as f:
        for rid, r in reqs.items():
            msgs = r["messages"]
            assert [m["role"] for m in msgs] == ["system", "user", "assistant"], rid
            prompt = BOS + msgs[0]["content"] + USER + msgs[1]["content"] + ASSISTANT + NO_THINK
            sys_at = len(BOS)
            user_at = sys_at + len(msgs[0]["content"]) + len(USER)
            starts[rid] = {"system": (0, sys_at), "user": (1, user_at), "assistant": (2, len(prompt))}
            f.write(json.dumps({"id": f"{ENV}/{r['run']}:{r['episode']}", "messages": msgs}) + "\n")
    stats: Counter = Counter()
    with (out / "reads.jsonl").open("w") as f:
        for row in map(json.loads, (a.run_dir / a.reads).read_text().splitlines()):
            r = reqs[row["id"]]
            msg_index, start = starts[row["id"]][row["role"]]
            content = r["messages"][msg_index]["content"]
            offset = row["char"] - start
            if 0 <= offset < len(content) and content[offset : offset + len(row["token"])] == row["token"]:
                stats["placed"] += 1
            else:  # BOS, <｜User｜>, <｜Assistant｜>, </think>: not in any message's content
                offset = None
                stats["template"] += 1
            bullets = [{"text": row[s], "fve": None, "coef": None} for s in SAMPLES if row.get(s)]
            f.write(
                json.dumps(
                    {
                        "id": f"{ENV}/{r['run']}:{r['episode']}",
                        "token_index": row["pos"],
                        "token": row["token"],
                        "prefix": row["ctx_left"],
                        "msg_index": msg_index,
                        "role": row["role"],
                        "field": "content",
                        "char_offset": offset,
                        "layer": row["layer"],
                        "bullets": bullets,
                        "joint_fve": None,
                    }
                )
                + "\n"
            )
            stats[f"role={row['role']}"] += 1
    print(f"{len(reqs)} requests -> {out}; reads {dict(stats)}")


if __name__ == "__main__":
    main()
