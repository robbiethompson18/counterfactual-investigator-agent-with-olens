"""O-lens read requests for the alignment_faking v8x prompts (2026-10-29): one transcript per v8x arm on item q0, read at
every token of the system prompt and of the user turn except the AdvBench request itself (the part every item shares),
plus the assistant header and empty think block.

    uv run --with transformers --with jinja2 python scripts/af_v8x_olens_requests.py qwen OUT.jsonl   # olens_read_requests.py
    uv run python scripts/af_v8x_olens_requests.py dsv4 OUT.jsonl --run data/envs/alignment_faking/2026-10-29_v8x-dsv4flash-nocot
        # dsv4_olens/vllm_capture_read.py --shared-prompt (positions from each row's `query`)
    uv run --with transformers --with jinja2 python scripts/af_v8x_olens_requests.py glm OUT.jsonl --prompts v8n-,v10
        # glm_qwen_reader/ (GLM-5.3 lens + the Qwen reader of GLM's block 51)

glm: the raw empty-think prompt GLM 5.3 ran under (render.glm_raw_prompt: "[gMASK]<sop><|system|>Reasoning Effort: Max", the
turns, "<|assistant|><think></think>"), read at every token from index 2 (Celeste's rule: not [gMASK]<sop>) except the request.
A position whose token prefix an earlier row already reads is read once and goes in `dups` as [pos, canon_id, canon_pos]. Rows
carry prompt_ids, the token strings (decode of each id) and chat sections, so glm_qwen_reader/to_reads.py needs no tokenizer.
"""

import argparse
import json
from pathlib import Path

from environments.alignment_faking import env

ap = argparse.ArgumentParser()
ap.add_argument("model", choices=["qwen", "dsv4", "glm"])
ap.add_argument("out")
ap.add_argument("--item", default="q0")
ap.add_argument("--prompts", default="v8x-", help="comma list of prompt names or name prefixes (e.g. v8n-,v10)")
ap.add_argument("--run", help="dsv4: the run folder whose episode on --item each row reuses (its reply comes after every read)")
args = ap.parse_args()
query = env.QUERIES[int(args.item[1:])]
arms = [a for a in env.ARMS if any(env.ARM_FACTORS[a]["prompt"].startswith(p) for p in args.prompts.split(","))]
rows = []
if args.model == "qwen":
    from transformers import AutoTokenizer

    from counterfactual_investigator_agent_with_olens.render import render_request

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    for arm in arms:
        ep = env.build(arm, args.item)
        msgs = [*ep.messages, {"role": "assistant", "content": "<rejected/>"}]  # after every read position, so it can't change a read
        row = {
            "id": f"alignment_faking/{env.ARM_FACTORS[arm]['prompt'].split('-')[0]}:{arm}~{args.item}",
            "env": "alignment_faking",
            "arm": arm,
            "messages": msgs,
        }
        _, ids, where = render_request(tok, row)
        user = msgs[1]["content"].strip()
        q0, q1 = user.index(query), user.index(query) + len(query)
        reply_tok = next(i for i, w in enumerate(where) if w[0] == 2 and w[1] == "content")
        row["positions"] = [i for i, w in enumerate(where[:reply_tok]) if not (w[0] == 1 and w[1] == "content" and q0 <= w[2] < q1)]
        rows.append(row)
elif args.model == "glm":
    from transformers import AutoTokenizer

    from counterfactual_investigator_agent_with_olens.render import glm_raw_prompt

    tok = AutoTokenizer.from_pretrained("zai-org/GLM-5.3")
    seen = {}  # token prefix -> (row id, position) of its first read
    n_own = n_dup = 0
    for arm in arms:
        ep = env.build(arm, args.item)
        text = glm_raw_prompt(ep.messages)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        toks = [tok.decode([t]) for t in ids]
        q0 = text.rindex(query)
        q1 = q0 + len(query)
        u, a = (toks.index(m) for m in ("<|user|>", "<|assistant|>"))
        rid = f"alignment_faking/{env.ARM_FACTORS[arm]['prompt'].split('-')[0]}:{arm}~{args.item}"
        own, dups = [], []
        for i, (s, e) in enumerate(offs):
            if i < 2 or q0 <= s < e <= q1:
                continue
            key = tuple(ids[: i + 1])
            if key in seen:
                dups.append([i, *seen[key]])
            else:
                seen[key] = (rid, i)
                own.append(i)
        n_own, n_dup = n_own + len(own), n_dup + len(dups)
        rows.append(
            {
                "id": rid,
                "env": "alignment_faking",
                "arm": arm,
                "prompt": env.ARM_FACTORS[arm]["prompt"],
                "item": args.item,
                "query": query,
                "prompt_ids": ids,
                "tokens": toks,
                "sections": [["system", 0, u], ["user", u, a], ["assistant", a, len(ids)]],
                "positions": own,
                "dups": dups,
            }
        )
    print(f"glm: {n_own} positions to read + {n_dup} shared-prefix copies")
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
