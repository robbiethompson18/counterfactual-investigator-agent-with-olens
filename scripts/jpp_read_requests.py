"""J++ lens reads (olens.jpp) of chat transcripts: the lens's top-k next tokens at every read position and source layer,
on the same engine, request format and pod recipe as the O-lens reader (scripts/olens_read_requests.py renders the rows
and picks the positions). Runs on a pod set up by `runpod/olens_reads.sh setup`:

    READER=scripts/jpp_read_requests.py runpod/olens_reads.sh run NAME REQUESTS.jsonl TAG [--layers all] [--positions all]

REQUESTS.jsonl is the O-lens reader's format; a row's `layers` overrides --layers for it (only --layers' maps are
loaded, so keep it a subset). Positions default to the O-lens ones (stop tokens and <|im_end|>, or a
row's `turn_starts`), so J++ rows join O-lens reads on (id, token_index); `--positions all` reads every token (the token
view of the viewer). OUT.jsonl gets one line per (transcript, position), appended per transcript (a rerun resumes):

    {id, token_index, token, msg_index, role, field, char_offset, layers: [...], top: [[[token, p] * k] per layer]}

the schema of runs/jpp_reads/2026-10-22_cfbench-spike-v1 (the HF transformers reader, deleted 2026-10-27), plus
OUT.provenance.json (lens file and revision, base revision, layers, k, positions). Throughput is the capture's: prefill
at ~8k tokens/s, the lens maps add ~0.2 ms per position for all 63 layers.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from olens.jpp import JPP, JPP_FILE, JPP_REPO
from olens.one_engine import OneEngine
from olens_read_requests import prepare, queue, resume, write_transcript  # scripts/ is sys.path[0] when this file is run as a script
from transformers import AutoTokenizer

from olens import provenance

DEFAULT_LAYERS = ",".join(str(L) for L in range(0, 61, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("requests")
    ap.add_argument("out")
    ap.add_argument("--layers", default=DEFAULT_LAYERS, help='source layers (residual after block L), or "all" for 0..62')
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--positions", default="stop", choices=("stop", "all"), help="the O-lens read positions, or every token")
    ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--vllm-mem", type=float, default=0.85, help="gpu_memory_utilization; the maps and lm_head (~6 GB) sit beside it")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--batch-positions", type=int, default=4096, help="positions captured per engine call")
    ap.add_argument("--chunk", type=int, default=512, help="positions per capture request (one prefill per chunk)")
    ap.add_argument("--shard", default="0/1", help="i/n: read the transcripts whose id hashes to i mod n")
    args = ap.parse_args()
    layers = list(range(63)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    out = Path(args.out)
    done = resume(out)
    print(
        f"{len(queue(args.requests, args.shard, done))} transcripts to read at {len(layers)} layers, positions={args.positions}", flush=True
    )

    tok = AutoTokenizer.from_pretrained(args.base)
    eng = OneEngine(tok, args.base, gpu_memory_utilization=args.vllm_mem, max_num_seqs=64, max_model_len=args.max_model_len)
    jpp = JPP(eng.base_dir, eng.norm_w, eng.eps, layers)
    out.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "script": Path(__file__).name,
                "lens": {"repo": JPP_REPO, "file": JPP_FILE, "revision": jpp.revision},
                "base": args.base,
                "base_revision": provenance.base_revision(),
                "engine": "vllm-metamodels one engine (capture only)",
                "layers": layers,
                "top_k": args.top_k,
                "positions": args.positions,
                "requests": Path(args.requests).name,
                "shard": args.shard,
            },
            indent=1,
        )
    )
    im_end, im_start = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|im_start|>")
    text = {}  # token id -> text, memoised (every position names top_k tokens per layer)
    t0, n_done, n_pos, secs = time.time(), 0, 0, {"capture": 0.0, "lens": 0.0}
    while todo := queue(args.requests, args.shard, done):
        jobs, reqs = [], []  # a job's positions go to the engine in chunks of --chunk, one capture request each
        for row in todo:
            try:
                msgs, ids, where, toks, positions, row_layers, _ = prepare(tok, row, layers, im_end, im_start)
            except json.JSONDecodeError as e:  # a tool call's `arguments` aren't JSON: vLLM 400s on it too, so Qwen never saw a rendering
                print(f"skip {row['id']}: tool call arguments aren't valid JSON ({e})", flush=True)
                done.add(row["id"])
                continue
            if args.positions == "all" and row.get("turn_starts") is None:
                positions = list(range(len(ids)))
            assert set(row_layers) <= set(layers), f"{row['id']}: layers {row_layers} not all in --layers"
            if len(ids) >= args.max_model_len:  # the engine would reject it; a rerun with a larger --max-model-len picks it up
                print(f"skip {row['id']}: {len(ids)} tokens >= --max-model-len {args.max_model_len}", flush=True)
                done.add(row["id"])
                continue
            chunks = [positions[a : a + args.chunk] for a in range(0, len(positions), args.chunk)]
            jobs.append({"row": row, "msgs": msgs, "ids": ids, "where": where, "toks": toks, "positions": positions,
                         "layers": row_layers, "reqs": range(len(reqs), len(reqs) + len(chunks))})  # fmt: skip
            reqs += [(ids, row_layers, c) for c in chunks]
            if sum(len(j["positions"]) for j in jobs) >= args.batch_positions:
                break
        ta = time.time()
        caps = eng.capture_many(reqs)
        tb = time.time()
        with out.open("a") as f:
            for j in jobs:
                write_transcript(out, j)
                acts = {L: torch.cat([caps[i][L] for i in j["reqs"]]) for L in j["layers"]}
                write(f, tok, j, j["layers"], jpp.top(acts, args.top_k), text)
                done.add(j["row"]["id"])
        secs["capture"] += tb - ta
        secs["lens"] += time.time() - tb
        n_done, n_pos = n_done + len(jobs), n_pos + sum(len(j["positions"]) for j in jobs)
        dt = time.time() - t0
        print(
            f"[{n_done}, {len(todo) - len(jobs)} queued] batch of {len(jobs)} transcripts, {sum(len(j['ids']) for j in jobs)} tokens, "
            f"{sum(len(j['positions']) for j in jobs)} positions | {n_pos / dt:.0f} positions/s, {n_done / dt * 3600:.0f} transcripts/h | "
            f"capture {secs['capture']:.0f}s lens {secs['lens']:.0f}s",
            flush=True,
        )


def write(f, tok, job, layers, top, text):
    """One line per read position of a transcript, the old HF reader's schema; `top` is olens.jpp.JPP.top's output."""
    for j, p in enumerate(job["positions"]):
        msg, field, off = job["where"][p]
        tops = [
            [[text.setdefault(t, tok.decode([t])), round(q, 4)] for t, q in zip(top[L][0][j].tolist(), top[L][1][j].tolist())]
            for L in layers
        ]
        rec = {
            "id": job["row"]["id"], "token_index": p, "token": job["toks"][p], "msg_index": msg,
            "role": job["msgs"][msg]["role"] if msg >= 0 else "template", "field": field, "char_offset": off,
            "layers": layers, "top": tops,
        }  # fmt: skip
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
