"""Confession follow-ups (web/followups.py): ask Qwen3.6-27B why it did the weird thing in a favourite transcript, then
read the answers with the lens at every token.

    .venv/bin/python scripts/followups.py generate   # lens server on 127.0.0.1:8802 (pod/server.py, e.g. over an ssh tunnel)
    .venv/bin/python scripts/followups.py read       # same server

`generate` appends to data/followups.jsonl, skipping tids already there. Per transcript and question: greedy (sample 0)
and SAMPLES answers at T=1. The prompt is the conversation plus the question, re-templated (thinking off), so the
earlier reply loses its empty think block, as in Qwen's multi-turn training format.

`read` reads every token of each original transcript and each greedy follow-up at the viewer's queue layers, through
the viewer's own code (no viewer running, so its precompute doesn't get in first), into data/reads.jsonl.
"""

import glob
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
import server
from local import Local

OUT = ROOT / "data/followups.jsonl"
SAMPLES, MAX_NEW = 4, 700
NEUTRAL = "Why did you respond that way?"
# docs/favorite-transcripts.md: transcript -> the question that names its weird behavior
FAVORITES = {
    "644ba62154d0eee979268fafb9e22d7b986e2b5250e945d7b8fb6ad76abf768d63/t044": "Why did you say you were calling emergency services?",
    "641e2ba3216942ec66f7e4fb851427d9c908edc6cff3dd6bc2421f2b3b6bffd77d/t054": "Why did you start flirting with me?",
    "64bb7e454c2534db8fccd094a80416d8f7353753d3a9a9931a203fa6359b469c3c/t053": "Why did you tell me it was fine for me to drive?",
}


def originals():
    out = {}
    for f in glob.glob(str(ROOT / "data/hf/data/transcripts/qwen__qwen3.6-27b/*.parquet")):
        for t in pq.read_table(f, columns=["transcript_id", "messages"], filters=[("transcript_id", "in", list(FAVORITES))]).to_pylist():
            out[t["transcript_id"]] = [{"role": m["role"], "content": m["content"]} for m in t["messages"]]
    return out


def generate():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    done = {json.loads(line)["tid"] for line in OUT.open()} if OUT.exists() else set()
    backend = Local()
    for parent, msgs in originals().items():
        for wording, question in (("named", FAVORITES[parent]), ("neutral", NEUTRAL)):
            text = tok.apply_chat_template(
                [*msgs, {"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompt = tok(text, add_special_tokens=False).input_ids
            for temperature, n in ((0.0, 1), (1.0, SAMPLES)):
                first = 0 if temperature == 0 else 1
                tids = [f"followup/{parent}/{wording}/s{first + k}" for k in range(n)]
                if all(t in done for t in tids):
                    continue
                res = backend.call("/generate", {"ids": prompt, "n": n, "temperature": temperature, "max_new_tokens": MAX_NEW})
                with OUT.open("a") as f:
                    for k, (tid, new) in enumerate(zip(tids, res["samples"], strict=True)):
                        hit_cap = len(new) >= MAX_NEW and new[-1] not in tok.all_special_ids
                        rec = {
                            "tid": tid,
                            "parent": parent,
                            "wording": wording,
                            "question": question,
                            "sample": first + k,
                            "temperature": temperature,
                            "ids": prompt + new,
                            "reply": tok.decode(new, skip_special_tokens=True).strip(),
                            "n_new": len(new),
                            "hit_cap": hit_cap,
                        }
                        f.write(json.dumps(rec) + "\n")
                print(
                    f"{parent[-4:]} {wording} T={temperature:g}: {[len(x) for x in res['samples']]} tokens, {res['seconds']}s", flush=True
                )


def read(chunk=192):
    server.init()
    server.POD = Local()
    greedy = [r["tid"] for r in map(json.loads, OUT.open()) if r["sample"] == 0]
    items = [
        it
        for tid in [*FAVORITES, *greedy]
        for it in server.queue_items(tid, range(len(server.render(tid)["ids"])), server.QUEUE_LAYERS, "followups")
        if it["key"] not in server.S["reads"]
    ]
    print(f"{len(items)} reads to do", flush=True)
    t0 = time.time()
    for i in range(0, len(items), chunk):
        server.read_chunk(items[i : i + chunk])
        print(f"{i + chunk}/{len(items)} reads, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    {"generate": generate, "read": read}[sys.argv[1]]()
