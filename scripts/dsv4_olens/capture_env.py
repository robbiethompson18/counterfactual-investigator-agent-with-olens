"""Capture V4 Flash block outputs at the read positions of agentic env transcripts (prep_overeager.py's requests), for
vllm_read.py's lens stage. Runs on the GPU box, with read.py's HF stack.

Each request is rendered with DeepSeek's own encoder (the model dir's encoding/encoding_dsv4.py, thinking mode, tools on the
system message, so every turn keeps its reasoning), as OpenRouter served it. Read positions are the Qwen reader's rule
ported to V4: every punctuation token (scripts/olens_read_requests.py's PUNCT) and every `</think>` and end-of-sentence
token, except inside tool results (file listings, CSVs, git logs: 58% of the positions on overeagerness), where only each
result's last token is read. A position whose prefix another transcript already has is captured once (shared system prompts, tool results).

Writes OUT/states.pt (rows: id, pos, key, token, role; one row per distinct prefix) and OUT/reads.transcripts.jsonl
(tokens, chat sections, request metadata; docs/olens-reads-format.md) plus OUT/positions.jsonl (id -> [pos, key]).

    python capture_env.py --model /data/v4flash --requests requests.jsonl --out out/ --layers 30
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch
from read import Capture

PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")  # scripts/olens_read_requests.py
ENDS = ("</think>", "<｜end▁of▁sentence｜>")
ROLE_MARK = re.compile(r"<｜User｜>|<｜Assistant｜>|<｜end▁of▁sentence｜>|<think>|</think>")


def to_v4(messages, tools):
    """Our episode messages (OpenAI layout, `reasoning` on assistant turns) -> encoding_dsv4's input."""
    out = []
    for m in messages:
        m = dict(m)
        if m["role"] == "assistant":
            m = {"role": "assistant", "content": m.get("content") or "", "reasoning_content": m.get("reasoning") or "",
                 "tool_calls": m.get("tool_calls") or None}  # fmt: skip
        elif m["role"] == "tool":
            m = {"role": "tool", "content": m.get("content") or "", "tool_call_id": m.get("tool_call_id", "")}
        out.append(m)
    if tools:
        assert out[0]["role"] == "system", "tools go on the system message"
        out[0] = out[0] | {"tools": tools}
    return out


def sections(text, offs):
    """[[name, start, end]] token spans: system, user, tool, reasoning, assistant (reply + tool calls)."""
    spans, cur, start = [], "system", 0
    for m in ROLE_MARK.finditer(text):
        tok = m.group()
        if tok == "<｜end▁of▁sentence｜>":  # ends an assistant turn; the next marker opens the next block
            continue
        nxt = {"<｜Assistant｜>": "assistant", "<think>": "reasoning", "</think>": "assistant"}.get(tok)
        nxt = nxt or ("tool" if text.startswith("<tool_result>", m.end()) else "user")
        spans.append((cur, start, m.start()))
        cur, start = nxt, m.start()
    spans.append((cur, start, len(text)))
    tok_of = lambda c: next((i for i, (s, e) in enumerate(offs) if e > c), len(offs))
    return [[n, tok_of(s), tok_of(e)] for n, s, e in spans if tok_of(e) > tok_of(s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="30")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="render + count positions only, no model")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.model) / "encoding"))
    from encoding_dsv4 import encode_messages
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",")]
    reqs = [json.loads(line) for line in Path(a.requests).read_text().splitlines()][: a.limit or None]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = None if a.dry else AutoModelForCausalLM.from_pretrained(a.model, device_map="auto", dtype="auto").eval()
    cap = None if a.dry else Capture(layers, model.model.layers)
    dev = None if a.dry else model.get_input_embeddings().weight.device
    seen, states, rows, total = set(), [], [], 0
    t = time.time()
    with open(out / "reads.transcripts.jsonl", "w") as ft, open(out / "positions.jsonl", "w") as fp, torch.no_grad():
        for qi, r in enumerate(reqs):
            text = encode_messages(to_v4(r["messages"], r["tools"]), thinking_mode="thinking")
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            pieces = [text[s:e] for s, e in offs]
            secs = sections(text, offs)
            tool_ends = {e - 1 for n, s, e in secs if n == "tool"}
            in_tool = {i for n, s, e in secs if n == "tool" for i in range(s, e - 1)}
            pos = [i for i, p in enumerate(pieces) if i in tool_ends or i not in in_tool and (PUNCT.match(p) or p in ENDS)]
            h, keys, want = hashlib.blake2b(digest_size=16), {}, set(pos)
            for i, x in enumerate(ids):  # prefix hash at each read position: equal prefix -> equal activation
                h.update(x.to_bytes(4, "little"))
                if i in want:
                    keys[i] = h.copy().hexdigest()
            new = [p for p in pos if keys[p] not in seen]
            if new and not a.dry:
                cap.want, cap.out = new, {}
                try:
                    model(torch.tensor([ids], device=dev), logits_to_keep=1)
                except torch.OutOfMemoryError:  # HF's eager attention on the longest transcripts (~20k tokens on 2x B200)
                    cap.want, cap.out = None, {}
                    torch.cuda.empty_cache()
                    print(f"OOM, skipped: {r['id']} ({len(ids)} tokens)", flush=True)
                    continue
                cap.want = None
                for j, p in enumerate(new):
                    states.append(torch.stack([cap.out[L][j] for L in layers]).to(torch.bfloat16))  # [n_layers, 4, 4096]
            seen.update(keys[p] for p in new)
            total += len(pos)
            meta = {k: r[k] for k in ("env", "arm", "item", "positive", "asked", "thinking") if k in r}
            ft.write(json.dumps({"id": r["id"], **meta, "tokens": pieces, "sections": secs}) + "\n")
            fp.write(json.dumps({"id": r["id"], "pos": [[p, keys[p]] for p in pos]}) + "\n")
            role = {i: n for n, s, e in secs for i in range(s, e)}
            rows += [{"id": r["id"], "pos": p, "key": keys[p], "token": pieces[p], "role": role[p], "arm": r["arm"]} for p in new]
            if qi % 20 == 0:
                print(f"{qi + 1}/{len(reqs)} {len(ids)} tokens, {len(pos)} positions ({len(new)} new); {len(rows)}/{total} so far, "
                      f"{time.time() - t:.0f}s", flush=True)  # fmt: skip
    print(f"{len(reqs)} transcripts, {total} read positions, {len(rows)} distinct prefixes, {time.time() - t:.0f}s", flush=True)
    if not a.dry:
        torch.save({"layers": layers, "states": torch.stack(states), "rows": rows}, out / "states.pt")


if __name__ == "__main__":
    main()
