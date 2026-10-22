"""O-lens reads at every stop token of chat transcripts: the agu18dec lens decoded by vLLM (olens.vllm_lens, merged
weights), scored with the AR (per-bullet and joint whitened FVE, as transcript-viewer/pod/server.py's score()).

Runs on a GPU pod in the vllm/vllm-openai image (one H200: vLLM takes ~half, the HF base for capture + AR the rest):

    PYTHONPATH=olens/src python scripts/olens_read_requests.py REQUESTS.jsonl OUT.jsonl MERGED_DIR [--layers 24,32,40,48,56]

REQUESTS.jsonl: one transcript per line, {id, messages (OpenAI chat format; assistant turns may carry reasoning_content
and tool_calls), tools ("evalcue.harness.BASH_TOOL" or a list of tool schemas), priority?, env?, arm?}. Rows are read in
priority order, round-robin over (env, arm) within a priority. Rendered with Qwen3.6-27B's chat template, as vLLM
renders a chat request (tool-call arguments parsed to dicts), so the tokens are the ones Qwen saw.

Positions: every token matching the viewer's PUNCT regex (transcript-viewer/web/server.py) and every <|im_end|>,
prompt included. OUT.jsonl gets one line per (transcript, position, layer), appended per transcript; transcripts
already in OUT are skipped, so a rerun resumes:

    {id, token_index, token, prefix (decoded text of the 32 tokens up to and including this one), msg_index (-1:
     template/tool definitions), role, field (content / reasoning_content / tool_call / null), char_offset (into that
     field's text, or null), layer, bullets: [{text, fve, coef}] (sorted by fve), joint_fve}
"""

import argparse
import copy
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from olens.ar import AR
from olens.lens import Sampling, load_base
from olens.readout import Readout, bullets
from olens.vllm_lens import VllmLens
from transformers import AutoTokenizer

PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")  # transcript-viewer/web/server.py
BASH_TOOL = {  # evalcue/harness.py; copied so this runs on a pod without evalcue's deps
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the working directory. Returns stdout+stderr (truncated) and the exit code.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}
SAMPLING = Sampling(max_new_tokens=160, max_bullets=4)  # greedy, 4 bullets: the viewer's read
PREFIX = 32


def render(tok, row):
    """-> (text, token ids, char spans, per-token (msg_index, field, char_offset))."""
    msgs = copy.deepcopy(row["messages"])
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            if isinstance(tc["function"]["arguments"], str):
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    tools = [BASH_TOOL] if row.get("tools") == "evalcue.harness.BASH_TOOL" else row.get("tools")
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    # Find each message's fields in the rendered text, in order; a token belongs to the last message whose first
    # field starts at or before it (so template tokens around a message go with it), -1 before the first.
    fields, cur = [], 0  # (start, end, msg_index, field)
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
def capture(model, ids, layers, positions, chunk=4096):
    """Residual after block L at `positions`, for each L in layers: {L: [P, d] fp32 CPU}. Hooks keep only those rows.
    Prefilled `chunk` tokens at a time through the cache: one forward over a 15k-token transcript needed >16 GB of
    transients next to vLLM and ran out of memory; 2048-token chunks peak at 51.7 GiB (weights ~50). Chunking moves
    the residual by bf16 noise (256-token chunks vs one forward: mean cos 0.997-0.99998, min 0.958 at L56)."""
    got = {L: [] for L in layers}
    cur = {}  # this chunk's positions, relative to the chunk

    def keep(L):
        def hook(mod, inp, out):
            got[L].append((out[0] if isinstance(out, tuple) else out)[0, cur["pos"]].float().cpu())

        return hook

    hooks = [model.model.layers[L].register_forward_hook(keep(L)) for L in layers]
    past = None
    try:
        for s in range(0, len(ids), chunk):
            cur["pos"] = torch.tensor([p - s for p in positions if s <= p < s + chunk], dtype=torch.long, device=model.device)
            out = model.model(input_ids=torch.tensor([ids[s : s + chunk]], device=model.device), past_key_values=past, use_cache=True)
            past = out.past_key_values
    finally:
        for h in hooks:
            h.remove()
    return {L: torch.cat(v) for L, v in got.items()}


@torch.no_grad()
def score(tok, ar, readout, texts, vecs, layers):
    """As transcript-viewer/pod/server.py's score(): per-bullet FVE (cos^2 in the whitened space, 0 if negative) and the
    joint FVE of the NNLS refit on all bullets. Batched: whitening is one matmul per layer and only each read's small
    Gram matrix leaves the GPU (a per-read loop with a 5120-dim scipy NNLS each took 40% of the run)."""
    bls = [bullets(t) for t in texts]
    flat = [(x, k) for k, bl in enumerate(bls) for x in bl]
    if not flat:
        return [([], 0.0) for _ in texts]
    spans = [tok(x, add_special_tokens=False).input_ids[:32] for x, _ in flat]
    lays = [layers[k] for _, k in flat]
    D = torch.cat([ar(spans[i : i + 256], lays[i : i + 256]) for i in range(0, len(spans), 256)])
    Z = torch.empty(len(texts), vecs.shape[1], device=D.device)
    owner = torch.tensor([k for _, k in flat], device=D.device)
    for L in set(layers):
        W = readout.whitener(L)
        ks = torch.tensor([k for k in range(len(texts)) if layers[k] == L], device=D.device)
        Z[ks] = W(vecs[ks.cpu()].to(D.device))
        js = torch.isin(owner, ks)
        D[js] = W(D[js])
    # [reads, K, d] of each read's whitened bullet directions, zero-padded; K = most bullets in any read
    K = max(map(len, bls))
    first = [0]
    for bl in bls:
        first.append(first[-1] + len(bl))
    idx = torch.tensor([[first[k] + i if i < len(bl) else len(flat) for i in range(K)] for k, bl in enumerate(bls)], device=D.device)
    P = torch.cat([D, torch.zeros(1, D.shape[1], device=D.device)])[idx]
    G = (P @ P.transpose(1, 2)).double().cpu().numpy()
    b = (P @ Z[:, :, None])[..., 0].double().cpu().numpy()
    zz = (Z * Z).sum(-1).double().cpu().numpy()
    out = []
    for k, bl in enumerate(bls):
        n = len(bl)
        if not n:
            out.append(([], 0.0))
            continue
        joint, c = nnls_gram(G[k, :n, :n], b[k, :n], zz[k])
        single = np.clip(b[k, :n] / np.sqrt(np.diag(G[k, :n, :n]) * zz[k]), 0, None) ** 2
        bs = [{"text": x, "fve": round(float(single[i]), 4), "coef": round(float(c[i]), 4)} for i, x in enumerate(bl)]
        bs.sort(key=lambda x: -x["fve"])
        out.append((bs, round(float(joint), 4)))
    return out


def nnls_gram(G, b, zz):
    """olens.fve.fve_nnls in Gram form: min ||z - sum c_i d_i||^2 over c >= 0, with G = D D^T, b = D z, zz = z.z. Exact
    for a few directions: the optimum is the least-squares fit on some subset with every coefficient >= 0, where the
    squared residual is zz - b_S . c_S. Returns (FVE, c)."""
    k = len(b)
    best, best_c = zz, np.zeros(k)
    for mask in range(1, 2**k):
        S = [i for i in range(k) if mask >> i & 1]
        cS = np.linalg.pinv(G[np.ix_(S, S)]) @ b[S]
        if (cS >= 0).all() and zz - b[S] @ cS < best:
            best, best_c = zz - b[S] @ cS, np.zeros(k)
            best_c[S] = cS
    return 1.0 - best / zz, best_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("requests")
    ap.add_argument("out")
    ap.add_argument("merged")
    ap.add_argument("--layers", default="24,32,40,48,56")
    # vLLM's share of the GPU; the HF base (capture + AR) needs ~60 GB of the rest. Each concurrent decode needs a
    # Mamba (linear-attention state) block: 0.45 of an H200 fits only 139, 0.55 fits 256.
    ap.add_argument("--vllm-mem", type=float, default=0.55)
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--shard", default="0/1", help="i/n: read every n-th transcript (in priority order) from the i-th")
    ap.add_argument("--skip-ids", help="file of transcript ids to skip, one per line (e.g. read by another shard)")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    rows = [json.loads(line) for line in Path(args.requests).read_text().splitlines()]
    # priority order; within a priority, round-robin over (env, arm) so early results cover every case
    rank, seen = {}, {}
    for r in rows:
        g = (r.get("priority", 0), r.get("env"), r.get("arm"))
        rank[r["id"]] = (g[0], seen.get(g, 0))
        seen[g] = seen.get(g, 0) + 1
    rows.sort(key=lambda r: rank[r["id"]])
    out = Path(args.out)
    i, n = map(int, args.shard.split("/"))
    rows = rows[i::n]
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    done |= set(Path(args.skip_ids).read_text().split()) if args.skip_ids else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(todo)} of {len(rows)} transcripts to read at layers {layers}", flush=True)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    lens = VllmLens(tok, args.merged, gpu_memory_utilization=args.vllm_mem, max_num_seqs=args.max_seqs)  # before the HF model
    _, model = load_base()
    lens.model = model
    ar = AR(model)
    readout = Readout(lens, ar)
    for L in layers:
        readout.whitener(L)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    t0, n_reads, secs = time.time(), 0, dict.fromkeys(("capture", "decode", "score"), 0.0)
    for n, row in enumerate(todo, 1):
        ta = time.time()
        msgs, ids, where = render(tok, row)
        toks = [tok.decode([i]) for i in ids]
        positions = [i for i, x in enumerate(ids) if x == im_end or PUNCT.match(toks[i])]
        acts = capture(model, ids, layers, positions)
        tb = time.time()
        items = [(p, L, acts[L][k]) for k, p in enumerate(positions) for L in layers]
        vecs = torch.stack([v for _, _, v in items])
        texts = [t[0] for t in lens.read_items(vecs, [L for _, L, _ in items], SAMPLING)]
        tc = time.time()
        scored = score(tok, ar, readout, texts, vecs, [L for _, L, _ in items])
        for k, dt in (("capture", tb - ta), ("decode", tc - tb), ("score", time.time() - tc)):
            secs[k] += dt
        with out.open("a") as f:
            for (p, L, _), (bs, joint) in zip(items, scored):
                msg, field, off = where[p]
                rec = {
                    "id": row["id"], "token_index": p, "token": toks[p], "prefix": tok.decode(ids[max(0, p - PREFIX + 1) : p + 1]),
                    "msg_index": msg, "role": msgs[msg]["role"] if msg >= 0 else "template", "field": field, "char_offset": off,
                    "layer": L, "bullets": bs, "joint_fve": joint,
                }  # fmt: skip
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_reads += len(items)
        dt = time.time() - t0
        stages = " ".join(f"{k} {v:.0f}s" for k, v in secs.items())
        print(f"[{n}/{len(todo)}] {row['id']} p{row.get('priority')}: {len(ids)} tokens, {len(items)} reads | "
              f"{n_reads / dt:.1f} reads/s, {n / dt * 3600:.0f} transcripts/h | {stages}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
