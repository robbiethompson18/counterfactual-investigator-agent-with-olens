"""O-lens read requests for the alignment_faking v8x prompts (2026-10-29): one transcript per v8x arm on item q0, read at
every token of the system prompt and of the user turn except the AdvBench request itself (the part every item shares),
plus the assistant header and empty think block. Qwen3.6-27B rendering (scripts/olens_read_requests.py's input).

    uv run --with transformers python scripts/af_v8x_olens_requests.py OUT.jsonl [--item q0]
"""

import argparse
import json

from transformers import AutoTokenizer

from counterfactual_investigator_agent_with_olens.render import render_request
from environments.alignment_faking import env

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--item", default="q0")
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
query = env.QUERIES[int(args.item[1:])]
n = 0
with open(args.out, "w") as f:
    for arm in (a for a in env.ARMS if a.startswith("v8x-")):
        ep = env.build(arm, args.item)
        msgs = [*ep.messages, {"role": "assistant", "content": "<rejected/>"}]  # after every read position, so it can't change a read
        row = {"id": f"alignment_faking/v8x:{arm}~{args.item}", "env": "alignment_faking", "arm": arm, "messages": msgs}
        _, ids, where = render_request(tok, row)
        user = msgs[1]["content"].strip()
        q0, q1 = user.index(query), user.index(query) + len(query)
        reply_tok = next(i for i, w in enumerate(where) if w[0] == 2 and w[1] == "content")
        row["positions"] = [i for i, w in enumerate(where[:reply_tok]) if not (w[0] == 1 and w[1] == "content" and q0 <= w[2] < q1)]
        n += len(row["positions"])
        f.write(json.dumps(row) + "\n")
print(f"{n} positions over 32 arms -> {args.out}")
