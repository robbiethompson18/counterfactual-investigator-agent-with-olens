"""Decapitated-network unroll: Qwen3.6-27B cut after block L, with the J++ lens as its head, generating text on its own.

The cut model is blocks 0..L, then next-token logits = lm_head(norm(J[L] h_L)) (olens.jpp's read, Ayonrinde & Lindsey 2026,
neuronpedia/jacobian-lens). It is a smaller language model: each sampled token is appended and blocks 0..L run on it, so
the first token of a readout is the J++ read at the prompt's last token and the rest is what the cut model says next.
Layer "full" is the control: all 64 blocks and the model's own head, i.e. Qwen continuing the prompt.

Runs on a pod with HF transformers (the vllm image has it; `runpod/CLAUDE.md`), one GPU with 80 GB+:

    python3 scripts/decap_unroll.py requests.jsonl readouts.jsonl [--layers 20,24,...,60,full] [--steps 32] [--limit N]

requests.jsonl: {family, id, prompt}, plain text, no chat template, no special tokens (WorkspaceBench's `plain` render).
readouts.jsonl, one line per (item, layer):

    {family, id, layer, token, top10: [[token, p] * 10], greedy: {text, tokens}, norepeat: {...}, sample: {...}}

`top10` is the head's distribution at the prompt's last token (the static J++ read); the three decodes are greedy, greedy
with every generated token banned from repeating, and one sample at WorkspaceBench's settings (T 1, top-k 64, top-p 0.95).
Prompts are batched by exact token length, so nothing is padded (Qwen3.6's linear-attention blocks carry state across
positions). The first batch of each layer checks the cached decode against a no-cache forward pass.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.6-27B"
JPP_REPO, JPP_FILE = "neuronpedia/jacobian-lens", "qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt"
VARIANTS = ("greedy", "norepeat", "sample")
TOP_K, TOP_P = 64, 0.95


def sample(logits, gen):
    """One draw per row at T 1 from the top-64 tokens, cut to the 0.95 nucleus."""
    v, i = logits.topk(TOP_K)
    p = v.softmax(-1)
    p = p * (p.cumsum(-1) - p < TOP_P)
    return i.gather(1, torch.multinomial(p, 1, generator=gen)).squeeze(1)


@torch.no_grad()
def unroll(body, head, ids, steps, gen, check=False):
    """ids [B, n] -> (generated ids [B * 3, steps], rows ordered greedy/norepeat/sample per prompt; top-10 (ids, p) of the
    head at the prompt's last token, [B, 10])."""
    x = ids.repeat_interleave(len(VARIANTS), 0)
    banned = torch.zeros(len(x), head.vocab, dtype=torch.bool, device=x.device)
    past, new, out, top = None, x, [], None
    for t in range(steps):
        o = body(input_ids=new, past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = head(o.last_hidden_state[:, -1])
        if t == 0:
            top = logits[:: len(VARIANTS)].softmax(-1).topk(10)
        nxt = logits.argmax(-1)
        nxt[1::3] = logits[1::3].masked_fill(banned[1::3], float("-inf")).argmax(-1)
        nxt[2::3] = sample(logits[2::3], gen)
        banned.scatter_(1, nxt[:, None], True)
        out.append(nxt)
        new = nxt[:, None]
    out = torch.stack(out, 1)
    if check:  # the cached decode's last step against one forward pass over the whole sequence
        ref = head(body(input_ids=torch.cat([x, out[:, :-1]], 1), use_cache=False).last_hidden_state[:, -1])
        agree = (ref.argmax(-1)[::3] == out[::3, -1]).float().mean().item()
        print(f"  cache check: greedy top-1 agreement {agree:.2f}, max |dlogit| {(ref - logits).abs().max().item():.2f}", flush=True)
        assert agree > 0.9, "cached decode disagrees with the no-cache forward pass"
    return out, top


class Head:
    """lm_head(norm(J h)) for a cut model, lm_head(norm(h)) for the full one."""

    def __init__(self, norm, lm_head, J=None):
        self.norm, self.lm_head, self.J, self.vocab = norm, lm_head, J, lm_head.weight.shape[0]

    def __call__(self, h):
        if self.J is not None:
            h = (h.float() @ self.J.T).to(h.dtype)
        return self.lm_head(self.norm(h)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("requests")
    ap.add_argument("out")
    grid = ",".join(str(L) for L in range(20, 61, 4))
    ap.add_argument("--layers", default=grid + ",full", help='cut after these blocks (WorkspaceBench\'s grid); "full" = no cut')
    ap.add_argument("--steps", type=int, default=32, help="tokens generated per readout")
    ap.add_argument("--batch", type=int, default=64, help="prompts per batch (x3 decodes)")
    ap.add_argument("--limit", type=int, default=0, help="first N requests only (smoke)")
    args = ap.parse_args()
    layers: list = [x if x == "full" else int(x) for x in args.layers.split(",")]

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda").eval()
    body, blocks, norm = model.model, model.model.layers, model.model.norm
    body.norm = nn.Identity()  # body() now returns the residual after its last block, not the normed one
    maps = torch.load(hf_hub_download(JPP_REPO, JPP_FILE), map_location="cpu", weights_only=True)["J"]

    rows = [json.loads(line) for line in Path(args.requests).read_text().splitlines()]
    rows = rows[: args.limit] if args.limit else rows
    by_len = defaultdict(list)
    for r in rows:
        r["ids"] = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        by_len[len(r["ids"])].append(r)
    batches = [g[a : a + args.batch] for g in by_len.values() for a in range(0, len(g), args.batch)]
    print(f"{len(rows)} prompts in {len(batches)} same-length batches, layers {layers}, {args.steps} steps", flush=True)

    one = lambda i: tok.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    gen = torch.Generator("cuda").manual_seed(0)
    with open(args.out, "w") as f:
        for L in layers:
            t0 = time.time()
            body.layers = blocks if L == "full" else nn.ModuleList(blocks[: L + 1])
            head = Head(norm, model.lm_head, None if L == "full" else maps[L].to("cuda", torch.float32))
            for b, batch in enumerate(batches):
                ids = torch.tensor([r["ids"] for r in batch], device="cuda")
                out, top = unroll(body, head, ids, args.steps, gen, check=b == 0)
                for k, r in enumerate(batch):
                    row: dict = {"family": r["family"], "id": r["id"], "layer": L, "token": one(r["ids"][-1])}
                    row["top10"] = [[one(i), round(p, 4)] for i, p in zip(top.indices[k].tolist(), top.values[k].tolist(), strict=True)]
                    for v, name in enumerate(VARIANTS):
                        g = out[k * len(VARIANTS) + v].tolist()
                        row[name] = {"text": tok.decode(g, skip_special_tokens=False, clean_up_tokenization_spaces=False)}
                        row[name]["tokens"] = [one(i) for i in g]
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            print(f"layer {L}: {time.time() - t0:.0f}s | {r['prompt'][-60:]!r} -> {row['norepeat']['text'][:80]!r}", flush=True)
    Path(args.out).with_suffix(".done").touch()


if __name__ == "__main__":
    main()
