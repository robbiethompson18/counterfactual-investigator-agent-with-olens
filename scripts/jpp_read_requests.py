"""J++ lens reads at every token of chat transcripts, at every 4th layer (olens.extras.JPP: the neuronpedia/jacobian-lens
qwen3.6-27b/jpp lens, logits at layer l = lm_head(norm(h_l @ J[l].T)), h_l the output of block l).

The J++ counterpart of scripts/nla_read_requests.py: same requests, render() and order as the O-lens reader, so rows join the O-lens
and NLA rows on (id, token_index). Those read only stop tokens; this reads every token. Runs on one 80 GB GPU (base ~54 GB
+ 16 J matrices ~0.8 GB), from the repo root:

    PYTHONPATH=olens/src python scripts/jpp_read_requests.py REQUESTS.jsonl OUT.jsonl [--shard i/n] [--chunk 4096]

REQUESTS.jsonl: as for the NLA reader. OUT.jsonl gets one line per (transcript, token), appended per transcript;
transcripts already in OUT are skipped, so a rerun resumes:

    {id, token_index, token, msg_index (-1: template/tool definitions), role, field (content / reasoning_content /
     tool_call / null), char_offset, layers ([0, 4, ..., 60]), top ([[decoded_token, prob] * 5] per layer)}

Probabilities are the full-vocab softmax, rounded to 4 places.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from olens.extras import JPP
from olens.lens import load_base
from olens_read_requests import render

LAYERS, K = list(range(0, 61, 4)), 5


@torch.no_grad()
def read(model, J, ids, chunk=4096, rows=2048):
    """-> (top-K token ids [T, len(LAYERS), K], probs [T, len(LAYERS), K]) at every position. The prefill goes `chunk`
    tokens at a time through the KV cache so long transcripts fit, as the NLA reader's capture()."""
    got = {}

    def hook(L):
        return lambda mod, inp, out: got.__setitem__(L, (out[0] if isinstance(out, tuple) else out)[0])

    handles = [model.model.layers[L].register_forward_hook(hook(L)) for L in LAYERS]
    idx, prob, past = [], [], None
    try:
        for s in range(0, len(ids), chunk):
            out = model.model(input_ids=torch.tensor([ids[s : s + chunk]], device=model.device), past_key_values=past, use_cache=True)
            past = out.past_key_values
            ci, cp = [], []  # per layer: [C, K]
            for li, L in enumerate(LAYERS):
                h, ti, tp = got[L], [], []
                for a in range(0, len(h), rows):
                    x = h[a : a + rows].float() @ J[li].T
                    lg = model.lm_head(model.model.norm(x.to(torch.bfloat16))).float()
                    top = lg.topk(K, dim=-1)
                    ti.append(top.indices)
                    tp.append((top.values - lg.logsumexp(-1, keepdim=True)).exp())
                ci.append(torch.cat(ti))
                cp.append(torch.cat(tp))
            idx.append(torch.stack(ci, 1).cpu())
            prob.append(torch.stack(cp, 1).cpu())
            got.clear()
    finally:
        for h in handles:
            h.remove()
    return torch.cat(idx).numpy(), torch.cat(prob).double().numpy().round(4)  # float64 first, or 0.1234 prints as 0.12340000271797180


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("requests")
    ap.add_argument("out")
    ap.add_argument("--chunk", type=int, default=4096, help="prefill tokens per forward")
    ap.add_argument("--shard", default="0/1", help="i/n: read every n-th transcript (in priority order) from the i-th")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.requests).read_text().splitlines()]
    rank, seen = {}, {}  # as nla_read_requests.py: priority order, round-robin over (env, arm) within a priority
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
    J = JPP(model, layers=LAYERS).J.float()  # [16, d, d]
    vocab = [tok.decode([t]) for t in range(len(tok))]
    vocab += [""] * (model.lm_head.out_features - len(vocab))  # padded lm_head rows
    print("loaded", flush=True)

    t0, n_tok = time.time(), 0
    for k, row in enumerate(todo, 1):
        msgs, ids, where = render(tok, row)
        top_i, top_p = read(model, J, ids, chunk=args.chunk)
        lines = []
        for p, (x, ti, tp) in enumerate(zip(ids, top_i.tolist(), top_p.tolist())):
            msg, field, off = where[p]
            rec = {
                "id": row["id"], "token_index": p, "token": vocab[x], "msg_index": msg,
                "role": msgs[msg]["role"] if msg >= 0 else "template", "field": field, "char_offset": off, "layers": LAYERS,
                "top": [[[vocab[t], q] for t, q in zip(a, b)] for a, b in zip(ti, tp)],
            }  # fmt: skip
            lines.append(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        with out.open("a") as f:
            f.write("".join(lines))
        n_tok += len(ids)
        dt = time.time() - t0
        print(f"[{k}/{len(todo)}] {row['id']} p{row.get('priority')}: {len(ids)} tokens | {n_tok / dt:.0f} tokens/s, "
              f"{k / dt * 3600:.0f} transcripts/h", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
