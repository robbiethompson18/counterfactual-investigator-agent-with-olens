"""O-lens read requests for the alignment_faking v8x prompts (2026-10-29): one transcript per v8x arm on item q0, read at
every token of the system prompt and of the user turn except the AdvBench request itself (the part every item shares),
plus the assistant header and empty think block.

    uv run --with transformers --with jinja2 python scripts/af_v8x_olens_requests.py qwen OUT.jsonl   # olens_read_requests.py
    uv run python scripts/af_v8x_olens_requests.py dsv4 OUT.jsonl --run data/envs/alignment_faking/2026-10-29_v8x-dsv4flash-nocot
        # dsv4_olens/vllm_capture_read.py --shared-prompt (positions from each row's `query`)
"""

import argparse
import json
from pathlib import Path

from environments.alignment_faking import env

ap = argparse.ArgumentParser()
ap.add_argument("model", choices=["qwen", "dsv4"])
ap.add_argument("out")
ap.add_argument("--item", default="q0")
ap.add_argument("--run", help="dsv4: the run folder whose episode on --item each row reuses (its reply comes after every read)")
args = ap.parse_args()
query = env.QUERIES[int(args.item[1:])]
arms = [a for a in env.ARMS if a.startswith("v8x-")]
rows = []
if args.model == "qwen":
    from transformers import AutoTokenizer

    from counterfactual_investigator_agent_with_olens.render import render_request

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    for arm in arms:
        ep = env.build(arm, args.item)
        msgs = [*ep.messages, {"role": "assistant", "content": "<rejected/>"}]  # after every read position, so it can't change a read
        row = {"id": f"alignment_faking/v8x:{arm}~{args.item}", "env": "alignment_faking", "arm": arm, "messages": msgs}
        _, ids, where = render_request(tok, row)
        user = msgs[1]["content"].strip()
        q0, q1 = user.index(query), user.index(query) + len(query)
        reply_tok = next(i for i, w in enumerate(where) if w[0] == 2 and w[1] == "content")
        row["positions"] = [i for i, w in enumerate(where[:reply_tok]) if not (w[0] == 1 and w[1] == "content" and q0 <= w[2] < q1)]
        rows.append(row)
else:
    eps = {}
    for f in sorted(Path(args.run).glob("episodes/*.jsonl")):
        for line in f.read_text().splitlines():
            e = json.loads(line)
            if e["item"] == args.item and e["sample"] == 0:
                eps[e["arm"]] = e
    for arm in arms:
        e = eps[arm]
        version, _, stage = arm.rpartition("/")
        msgs = [{"role": m["role"], "content": m["content"]} for m in e["messages"]]
        msgs[-1]["content"] = msgs[-1]["content"][:120]
        rows.append(
            {
                "id": f"{version}:{stage}:{args.item}",
                "run": Path(args.run).name,
                "episode": e["id"],
                "version": version,
                "arm": stage,
                "item": args.item,
                "sample": 0,
                "positive": bool(env.measure(e)["positive"]),
                "query": query,
                "messages": msgs,
            }
        )
Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "rows ->", args.out)
