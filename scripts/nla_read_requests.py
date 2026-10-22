"""NLA reads at every stop token of chat transcripts: the ceselder L42 natural-language autoencoder (olens.extras.NLA),
each explanation scored by its AR (olens.extras.NLACritic: cos and FVE against the L42 activation).

The NLA counterpart of scripts/olens_read_requests.py (the O-lens reader): same requests, same rendering, same
positions, so its rows join the O-lens rows on (id, token_index). Runs on one H200 (base ~54 GB + the AR, a 43-block copy
of the base, ~46 GB), from the repo root:

    PYTHONPATH=olens/src python scripts/nla_read_requests.py REQUESTS.jsonl OUT.jsonl [--batch 128] [--shard i/n]

REQUESTS.jsonl: as for the O-lens reader, {id, messages, tools ("evalcue.harness.BASH_TOOL" or a list), priority?, env?,
arm?}, read in priority order, round-robin over (env, arm) within a priority. FVE uses the viewer's calibration
(transcript-viewer/pod/calib_ids.json, as pod/server.py), so it means the same as on the O-lens viewer. OUT.jsonl gets one
line per (transcript, position), appended per transcript; transcripts already in OUT are skipped, so a rerun resumes:

    {id, token_index, token, prefix (the 32 tokens up to and including this one), msg_index (-1: template/tool
     definitions), role, field (content / reasoning_content / tool_call / null), char_offset, layer (42), text, cos, fve}
"""

import argparse
import copy
import json
import re
import time
from pathlib import Path

import torch
from olens.extras import NLA, NLACritic
from olens.lens import load_base

PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")  # transcript-viewer/web/server.py
BASH_TOOL = {  # evalcue/harness.py; copied so this runs on a pod without evalcue's deps
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the working directory. Returns stdout+stderr (truncated) and the exit code.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}
LAYER, PREFIX = 42, 32
CALIB = Path("transcript-viewer/pod/calib_ids.json")


def render(tok, row):
    """-> (messages, token ids, per-token (msg_index, field, char_offset)). Copied from scripts/olens_read_requests.py
    (the O-lens reader, not on main yet) so token indices match it exactly; merge the two once it lands."""
    msgs = copy.deepcopy(row["messages"])
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            if isinstance(tc["function"]["arguments"], str):
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    tools = [BASH_TOOL] if row.get("tools") == "evalcue.harness.BASH_TOOL" else row.get("tools")
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    fields, cur = [], 0  # (start, end, msg_index, field), in rendered order
    for i, m in enumerate(msgs):
        parts = [("reasoning_content", m.get("reasoning_content")), ("content", m.get("content"))]
        parts += [("tool_call", v) for tc in m.get("tool_calls") or [] for v in tc["function"]["arguments"].values()]
        for name, s in parts:
            s = s.strip() if isinstance(s, str) else None
            if not s:
                continue
            j = text.find(s, cur)
            if j < 0:
                continue
            fields.append((j, j + len(s), i, name))
            cur = j + len(s)
    starts = {}
    for a, _, i, _ in fields:
        starts.setdefault(i, a)
    where = []
    for a, _ in enc.offset_mapping:
        msg = max((i for i, s in starts.items() if s <= a), default=-1)
        f = next(((name, a - s) for s, e, i, name in fields if s <= a < e), (None, None))
        where.append((msg, *f))
    return msgs, enc.input_ids, where


@torch.no_grad()
def capture(model, ids, positions, chunk=4096):
    """Residual after block LAYER at `positions` ([P, d] fp32), prefilled `chunk` tokens at a time through the KV cache so
    long transcripts fit. Same as the O-lens reader's capture, for one layer."""
    got, cur = [], {}

    def hook(mod, inp, out):
        got.append((out[0] if isinstance(out, tuple) else out)[0, cur["pos"]].float().cpu())

    handle = model.model.layers[LAYER].register_forward_hook(hook)
    past = None
    try:
        for s in range(0, len(ids), chunk):
            cur["pos"] = torch.tensor([p - s for p in positions if s <= p < s + chunk], dtype=torch.long, device=model.device)
            out = model.model(input_ids=torch.tensor([ids[s : s + chunk]], device=model.device), past_key_values=past, use_cache=True)
            past = out.past_key_values
    finally:
        handle.remove()
    return torch.cat(got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("requests")
    ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=128, help="NLA explanations decoded at once")
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--shard", default="0/1", help="i/n: read every n-th transcript (in priority order) from the i-th")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.requests).read_text().splitlines()]
    rank, seen = {}, {}  # priority order; within a priority, round-robin over (env, arm) so early results cover every case
    for r in rows:
        g = (r.get("priority", 0), r.get("env"), r.get("arm"))
        rank[r["id"]] = (g[0], seen.get(g, 0))
        seen[g] = seen.get(g, 0) + 1
    rows.sort(key=lambda r: rank[r["id"]])
    i, n = map(int, args.shard.split("/"))
    out = Path(args.out)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    todo = [r for r in rows[i::n] if r["id"] not in done]
    print(f"{len(todo)} of {len(rows[i::n])} transcripts to read", flush=True)

    tok, model = load_base()
    nla, critic = NLA(tok, model), NLACritic(tok)
    calib = json.loads(CALIB.read_text())["ids"]
    critic.calibrate(torch.cat([capture(model, ids, list(range(4, len(ids)))) for ids in calib]))  # as pod/server.py
    print(f"loaded; FVE baseline {critic.base:.1f}", flush=True)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    t0, n_reads = time.time(), 0
    for k, row in enumerate(todo, 1):
        msgs, ids, where = render(tok, row)
        toks = [tok.decode([x]) for x in ids]
        positions = [p for p, x in enumerate(ids) if x == im_end or PUNCT.match(toks[p])]
        vecs = capture(model, ids, positions)
        texts = nla.explain(vecs, max_new_tokens=args.max_new_tokens, batch=args.batch)
        scores = critic.score(texts, vecs)
        with out.open("a") as f:
            for p, text, sc in zip(positions, texts, scores):
                msg, field, off = where[p]
                rec = {
                    "id": row["id"], "token_index": p, "token": toks[p], "prefix": tok.decode(ids[max(0, p - PREFIX + 1) : p + 1]),
                    "msg_index": msg, "role": msgs[msg]["role"] if msg >= 0 else "template", "field": field, "char_offset": off,
                    "layer": LAYER, "text": text, **sc,
                }  # fmt: skip
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_reads += len(positions)
        dt = time.time() - t0
        print(f"[{k}/{len(todo)}] {row['id']} p{row.get('priority')}: {len(ids)} tokens, {len(positions)} reads | "
              f"{n_reads / dt:.2f} reads/s, {k / dt * 3600:.0f} transcripts/h", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
