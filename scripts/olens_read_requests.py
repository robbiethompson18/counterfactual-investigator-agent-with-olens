"""O-lens reads at every stop token of chat transcripts, all on one vLLM engine (olens.one_engine: capture of the
residuals, the space lens through vLLM's LoRA runtime, the AR), scored per bullet and jointly with the whitened FVE
(as the viewer's lens server's score() did: transcript-viewer/pod/server.py, deleted 2026-10-26, at b842346).

Runs on a GPU pod set up by `runpod/olens_reads.sh setup` (vllm/vllm-openai:v0.27.1 + ceselder/vllm-metamodels, one H200):

    PYTHONPATH=olens/src python scripts/olens_read_requests.py REQUESTS.jsonl OUT.jsonl [--layers 24,32,40,48,56]

Many transcripts per step, so every engine call is one big batch (a decode step at batch 128 costs a third of one at
470, and the engine idles between calls): capture every new read position of up to --batch-reads reads in one generate,
decode them in one generate (one scheduler wave under max_num_seqs; two waves per generate measured 8% slower,
docs/olens-one-engine-reader.md), score every bullet in one AR generate, then write per transcript. Reads of a prefix
already seen (in the cache or earlier in the batch) are copied.

REQUESTS.jsonl: one transcript per line, {id, messages (OpenAI chat format; assistant turns may carry reasoning_content
and tool_calls), tools ("evalcue.harness.BASH_TOOL" or a list of tool schemas), priority?, env?, arm?}. Rows are read in
priority order, round-robin over (env, arm) within a priority. A row's optional `layers` overrides --layers for it
(whiteners are loaded for --layers only, so keep it a subset). Optional `generation_prompt: true` renders the open
assistant header after the last message (as at generation time), and `turn_starts: N` reads only at each <|im_start|>
and the N tokens after it instead of at stop tokens (e.g. N=3: <|im_start|>, "assistant", "\n", "<think>"), with
`last_turn_only: true` only at the last <|im_start|>, `turn_role: "assistant"` only at that role's headers; `positions:
"all"` reads every token. A row's
`shard_key` (default its id) picks its --shard, so rows sharing a prefix can share a process's read cache.
Rendered with Qwen3.6-27B's chat template, as vLLM
renders a chat request (tool-call arguments parsed to dicts), so the tokens are the ones Qwen saw.

Positions: every token matching the viewer's PUNCT regex (transcript-viewer/web/build.py) and every <|im_end|>,
prompt included. OUT.jsonl gets one line per (transcript, position, layer), appended per transcript; transcripts
already in OUT are skipped, so a rerun resumes:

    {id, token_index, token, prefix (decoded text of the 32 tokens up to and including this one), msg_index (-1:
     template/tool definitions), role, field (content / reasoning_content / tool_call / null), char_offset (into that
     field's text, or null), layer, bullets: [{text, fve, coef}] (sorted by fve), joint_fve}

plus OUT's provenance (olens.provenance: lens adapter, bullet prefix, AR, whitener, revisions, sampling) as
reads.provenance.json next to it, and each transcript's tokens and chat sections as OUT.transcripts.jsonl
(write_transcript), so expview shows the run with no tokenizer (docs/olens-reads-format.md).
"""

import argparse
import hashlib
import json
import re
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from olens.fve import Whitener
from olens.lens import Sampling, bullets
from olens.one_engine import OneEngine
from transformers import AutoTokenizer

from counterfactual_investigator_agent_with_olens.render import render_request, transcript_line  # PYTHONPATH=olens/src:src
from olens import provenance

PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")  # transcript-viewer/web/build.py
SAMPLING = Sampling(max_new_tokens=160, max_bullets=4)  # greedy, 4 bullets: the viewer's read
PREFIX = 32
AR_BATCH = 4096  # spans per AR generate; vLLM schedules the waves


@torch.no_grad()
def score(tok, lens, whiteners, texts, vecs, layers):
    """As the viewer lens server's score() (transcript-viewer/pod/server.py, deleted 2026-10-26, at b842346): per-bullet FVE
    (cos^2 in the whitened space, 0 if negative) and the joint FVE of the NNLS refit on all bullets. Batched: whitening is one
    matmul per layer and only each read's small Gram matrix leaves the GPU (a per-read loop with a 5120-dim scipy NNLS each
    took 40% of the run)."""
    bls = [bullets(t) for t in texts]
    flat = [(x, k) for k, bl in enumerate(bls) for x in bl]
    if not flat:
        return [([], 0.0) for _ in texts]
    spans = [tok(lens.bullet_prefix + x, add_special_tokens=False).input_ids[:32] for x, _ in flat]
    lays = [layers[k] for _, k in flat]
    D = torch.cat([lens.ar(spans[i : i + AR_BATCH], lays[i : i + AR_BATCH]) for i in range(0, len(spans), AR_BATCH)])
    Z = torch.empty(len(texts), vecs.shape[1], device=D.device)
    owner = torch.tensor([k for _, k in flat], device=D.device)
    for L in set(layers):
        W = whiteners[L]
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
    ap.add_argument("--layers", default="24,32,40,48,56")
    ap.add_argument("--base", default="Qwen/Qwen3.6-27B", help="the base checkpoint (HF id or dir)")
    ap.add_argument("--lens-lora", default="/workspace/lora_vllm/olens_space_rl600", help="lens adapter, keys renamed by lora_for_vllm.py")
    ap.add_argument("--ar-lora", default="/workspace/lora_vllm/ar_ptag_pooled", help="AR adapter with vLLM-renamed keys")
    ap.add_argument("--vllm-mem", type=float, default=0.88, help="gpu_memory_utilization")
    # each concurrent decode needs a linear-attention state block as well as KV; 1024 does not fit an H200 at 0.88
    ap.add_argument("--max-seqs", type=int, default=512, help="vLLM max_num_seqs")
    ap.add_argument("--max-model-len", type=int, default=32768, help="longest transcript the engine captures")
    ap.add_argument("--batch-reads", type=int, default=470, help="new reads per engine batch, under --max-seqs: one wave (2x: 8% slower)")
    ap.add_argument("--batch-transcripts", type=int, default=128, help="most transcripts per batch")
    ap.add_argument("--lora-mode", default="runtime", choices=("runtime", "merge"), help="vLLM LoRA kernels, or merge_lora per stage")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n: read the transcripts whose id hashes to i mod n")
    ap.add_argument("--skip-ids", help="file of transcript ids to skip, one per line (e.g. read by another shard)")
    ap.add_argument("--watch", action="store_true", help="when the queue is empty, re-read REQUESTS every minute")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out = Path(args.out)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    done |= set(Path(args.skip_ids).read_text().split()) if args.skip_ids else set()
    print(f"{len(queue(args.requests, args.shard, done))} transcripts to read at layers {layers}", flush=True)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    lens = OneEngine(tok, args.base, args.lens_lora, args.ar_lora, gpu_memory_utilization=args.vllm_mem,
                     max_num_seqs=args.max_seqs, max_model_len=args.max_model_len, lora_mode=args.lora_mode, debug=args.debug)  # fmt: skip
    whiteners = {L: Whitener(L, device="cuda", dtype=torch.float32) for L in layers}
    provenance.write(out, lens, layers, SAMPLING, __file__, requests=Path(args.requests).name, shard=args.shard)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    run_batched(args, tok, lens, whiteners, out, done, layers, im_end, im_start)


def prepare(tok, row, layers, im_end, im_start):
    """Render one request row: messages, token ids, per-token provenance, token strings, read positions, layers, prefix keys."""
    msgs, ids, where = render_request(tok, row)
    toks = [tok.decode([i]) for i in ids]
    positions = [i for i, x in enumerate(ids) if x == im_end or PUNCT.match(toks[i])]
    if row.get("turn_starts") is not None:  # only each <|im_start|> and the N tokens after it (the role, "\n", ...)
        starts = [i for i, x in enumerate(ids) if x == im_start]
        starts = starts[-1:] if row.get("last_turn_only") else starts
        if row.get("turn_role"):  # e.g. "assistant": only that role's headers
            starts = [i for i in starts if i + 1 < len(ids) and toks[i + 1] == row["turn_role"]]
        positions = sorted({j for i in starts for j in range(i, min(len(ids), i + 1 + row["turn_starts"]))})
    elif row.get("positions") == "all":
        positions = list(range(len(ids)))
    row_layers = row.get("layers") or layers  # a request row may name its own layers (a subset of --layers)
    return msgs, ids, where, toks, positions, row_layers, prefix_keys(ids, positions)


def run_batched(args, tok, lens, whiteners, out, done, layers, im_end, im_start):
    """The loop: many transcripts per step, every engine call one big batch (module docstring)."""
    t0, n_done, n_reads, secs = time.time(), 0, 0, dict.fromkeys(("capture", "decode", "score"), 0.0)
    cache = {}  # (prefix key, layer) -> scored read
    while True:
        todo = queue(args.requests, args.shard, done)
        if not todo:
            if not args.watch:
                break
            time.sleep(60)
            continue
        ta = time.time()
        jobs, want = [], {}  # want: (key, L) -> (job index, position), the first transcript in the batch that needs it
        for row in todo[: args.batch_transcripts]:
            msgs, ids, where, toks, positions, row_layers, keys = prepare(tok, row, layers, im_end, im_start)
            if len(ids) >= args.max_model_len:  # the engine would reject it; a rerun with a larger --max-model-len picks it up
                print(f"skip {row['id']}: {len(ids)} tokens >= --max-model-len {args.max_model_len}", flush=True)
                done.add(row["id"])
                continue
            new = [p for p in positions if any((keys[p], L) not in cache and (keys[p], L) not in want for L in row_layers)]
            for p in new:
                for L in row_layers:
                    if (keys[p], L) not in cache:
                        want.setdefault((keys[p], L), (len(jobs), p))
            jobs.append({"row": row, "msgs": msgs, "ids": ids, "where": where, "toks": toks, "keys": keys, "layers": row_layers,
                         "all": [(p, L) for p in positions for L in row_layers],
                         "new": sorted({p for (k, L), (j, p) in want.items() if j == len(jobs)})})  # fmt: skip
            if len(want) >= args.batch_reads:
                break
        caps = lens.capture_many([(j["ids"], j["layers"], j["new"]) for j in jobs])
        items = []  # (key, L, vec), one per decode
        for (k, L), (ji, p) in want.items():
            items.append((k, L, caps[ji][L][jobs[ji]["new"].index(p)]))
        vecs = torch.stack([v for _, _, v in items]) if items else torch.empty(0)
        item_layers = [L for _, L, _ in items]
        secs["capture"] += time.time() - ta
        tb = time.time()
        texts = [t[0] for t in lens.read_items(vecs, item_layers, SAMPLING)] if items else []
        secs["decode"] += time.time() - tb
        tc = time.time()
        scored = score(tok, lens, whiteners, texts, vecs, item_layers) if texts else []
        for (k, L, _), sc in zip(items, scored):
            cache[k, L] = sc
        for j in jobs:
            write(tok, out, j, [cache[j["keys"][p], L] for p, L in j["all"]])
            done.add(j["row"]["id"])
        secs["score"] += time.time() - tc
        n_done, n_reads = n_done + len(jobs), n_reads + len(items)
        dt = time.time() - t0
        stages = " ".join(f"{k} {v:.0f}s" for k, v in secs.items())
        n_all = sum(len(j["all"]) for j in jobs)
        n_tok = sum(len(j["ids"]) for j in jobs)
        print(f"[{n_done}, {len(todo) - len(jobs)} queued] batch of {len(jobs)} transcripts, {n_tok} tokens, {len(items)} reads "
              f"({n_all - len(items)} cached) | {n_reads / dt:.1f} reads/s, {n_done / dt * 3600:.0f} transcripts/h | {stages}",
              flush=True)  # fmt: skip


def prefix_keys(ids, positions):
    """position -> a hash of the token ids up to and including it (equal keys = equal prefixes, hence equal reads)."""
    h, keys, want = hashlib.blake2b(digest_size=16), {}, set(positions)
    for i, x in enumerate(ids):
        h.update(int(x).to_bytes(4, "little"))
        if i in want:
            keys[i] = h.copy().digest()
    return keys


def queue(path, shard, done):
    """REQUESTS minus `done`, this shard only: priority order, round-robin over (env, arm) within a priority so early
    results cover every case. Re-read on every call, so appending to REQUESTS adds work to a running reader."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    i, n = map(int, shard.split("/"))
    rank, seen = {}, {}
    for r in rows:
        g = (r.get("priority", 0), r.get("env"), r.get("arm"))
        rank[r["id"]] = (g[0], seen.get(g, 0))
        seen[g] = seen.get(g, 0) + 1
    rows = [r for r in rows if r["id"] not in done and zlib.crc32(r.get("shard_key", r["id"]).encode()) % n == i]
    return sorted(rows, key=lambda r: rank[r["id"]])


def write(tok, out, job, scored):
    ids, where, toks, msgs = job["ids"], job["where"], job["toks"], job["msgs"]
    write_transcript(out, job)
    with out.open("a") as f:
        for (p, L), (bs, joint) in zip(job["all"], scored):
            msg, field, off = where[p]
            rec = {
                "id": job["row"]["id"], "token_index": p, "token": toks[p], "prefix": tok.decode(ids[max(0, p - PREFIX + 1) : p + 1]),
                "msg_index": msg, "role": msgs[msg]["role"] if msg >= 0 else "template", "field": field, "char_offset": off,
                "layer": L, "bullets": bs, "joint_fve": joint,
            }  # fmt: skip
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_transcript(out, job):
    """The transcript's line in OUT.transcripts.jsonl (tokens as text, chat sections, request metadata): with OUT it is all
    a viewer needs (docs/olens-reads-format.md). Written before the reads, so a reread after a crash may repeat it."""
    with out.with_suffix(".transcripts.jsonl").open("a") as f:
        f.write(json.dumps(transcript_line(job["row"], job["msgs"], job["toks"], job["where"]), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
