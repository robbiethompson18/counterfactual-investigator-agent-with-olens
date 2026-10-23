"""Thought-Anchors attention reads on chat transcripts, for cfbench's `anchors` conditions (cfbench/anchors.py).

Thought Anchors (Bogdan, Macar, Nanda, Conmy 2025, arXiv 2506.19143) scores reasoning sentences two ways without
resampling: receiver heads (attention heads that concentrate on a few sentences) and attention suppression (mask
everything's view of one sentence, measure how much later tokens' predictions move). Both need only forward passes
over a transcript the model already wrote, so they fit the zero-counterfactual setting.

Runs on a GPU pod with the olens deps (one H200; Qwen3.6-27B in bf16 is ~54 GB):

    PYTHONPATH=olens/src:. python scripts/anchor_reads.py REQUESTS.jsonl OUT.jsonl [--check]

REQUESTS.jsonl is scripts/olens_read_requests.py's format, rendered the same way (so the tokens are the ones Qwen saw).
OUT.jsonl gets one line per transcript, appended; transcripts already in OUT are skipped:

    {id, n_tokens, layers (the full-attention layer of each row of `vertical`, 24 heads each),
     segments: [{tok_start, tok_end, msg_index, role, field, char_start, char_end (into the field's text, null for
                 template tokens), text, generated (model-written)}],
     vertical: [heads][segments] receiver score per head (null where no generated segment is >= 4 segments later),
     kl: [segments][segments] suppression effect: kl[s][u] = mean over u's model-written tokens of
         KL(normal || s suppressed) on the prediction of that token (null unless u is after s and model-written),
     seconds}

Segments: sentences and lines of each message field (a token ending in .!? before whitespace, or containing a
newline, closes one; at most MAX_SEG tokens), plus runs of template tokens (role/turn markers), which are never scored.

Receiver score (paper's definition): per head, average token attention into a segment x segment matrix
(A[j, i] = mean attention from j's tokens to i's tokens), then score i by the mean of A[j, i] over model-written
segments j at least 4 segments after i. Receiver heads are picked downstream (cfbench/anchors.py), by kurtosis.

Suppression on Qwen3.6-27B's hybrid stack (16 full-attention layers, 48 Gated DeltaNet layers): queries after segment s
can't attend to s's keys in the attention layers, and in every DeltaNet layer s's tokens neither write to nor decay the
recurrent state (beta -> 0, log-decay -> 0) and their q/k/v inputs to the short causal conv are zeroed, so no
information from s reaches later positions by any path. `--check` verifies that on the first transcript: with s
suppressed, replacing s's tokens with random ones must leave every later logit unchanged.
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from olens.lens import load_base
from transformers.integrations.sdpa_attention import repeat_kv
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from scripts.olens_read_requests import render

MAX_SEG = 96  # tokens per segment, so long tool outputs and code lines split
MIN_GAP = 4  # receiver score: only attention from segments at least this many segments later (the paper's cutoff)
NEG = -1e4  # pre-activation that sends DeltaNet's beta (sigmoid) and log-decay (softplus) to exactly 0
SENT_END = re.compile(r"[.!?][\"')\]*]*$")


def segments(tok, msgs, ids, where):
    """Token ranges [a, b) and their metadata; where[t] = (msg_index, field, char_offset) from render()."""
    toks = [tok.decode([i]) for i in ids]
    out, a = [], 0
    for t in range(len(ids)):
        nxt = t + 1
        end = nxt == len(ids) or where[nxt][:2] != where[t][:2] or nxt - a >= MAX_SEG
        if not end and toks[nxt].strip():  # absorb trailing whitespace tokens into this segment
            end = "\n" in toks[t] or (bool(SENT_END.search(toks[t].rstrip())) and toks[nxt][:1].isspace())
        if end:
            msg, field, c0 = where[a]
            c1 = where[t][2]
            role = msgs[msg]["role"] if msg >= 0 else "template"
            out.append({
                "tok_start": a, "tok_end": nxt, "msg_index": msg, "role": role, "field": field,
                "char_start": c0, "char_end": None if c1 is None else c1 + len(toks[t]),
                "text": "".join(toks[a:nxt]), "generated": role == "assistant" and field is not None,
            })  # fmt: skip
            a = nxt
    return out


class Patch:
    """Replaces sdpa in the full-attention layers with explicit attention in query blocks (so it can record attention
    and take a per-row suppression mask) and hooks the DeltaNet input projections. mode: None (plain attention),
    "receiver" (accumulate segment x segment attention sums per layer), "suppress" (row r of the batch suppresses the
    token span self.spans[r], or nothing if None)."""

    def __init__(self, model):
        self.mode, self.seg, self.n_seg, self.sums, self.spans = None, torch.zeros(0, dtype=torch.long), 0, {}, []
        self.orig = ALL_ATTENTION_FUNCTIONS["sdpa"]
        ALL_ATTENTION_FUNCTIONS["sdpa"] = self.attention
        self.attn_layers = [i for i, layer in enumerate(model.model.layers) if hasattr(layer, "self_attn")]
        for layer in model.model.layers:
            if hasattr(layer, "linear_attn"):
                la = layer.linear_attn
                la.in_proj_qkv.register_forward_hook(self.fill(0.0))
                la.in_proj_a.register_forward_hook(self.fill(NEG))
                la.in_proj_b.register_forward_hook(self.fill(NEG))

    def fill(self, value):
        def hook(mod, inp, out):
            if self.mode != "suppress":
                return out
            out = out.clone()
            for r, span in enumerate(self.spans):
                if span:
                    out[r, span[0] : span[1]] = value
            return out

        return hook

    def attention(self, module, query, key, value, attention_mask, scaling=None, **kwargs):
        B, H, Tq, _ = query.shape
        Tk = key.shape[2]
        assert Tq == Tk, "one forward over the whole transcript, no cache"
        key = repeat_kv(key, module.num_key_value_groups).float()
        value = repeat_kv(value, module.num_key_value_groups)
        out = torch.empty_like(query)
        kpos = torch.arange(Tk, device=query.device)
        block = max(1, 2048 // B)  # B * block * H * Tk fp32 logits: ~4 GB at 20k tokens
        for s in range(0, Tq, block):
            qpos = torch.arange(s, min(s + block, Tq), device=query.device)
            logits = (query[:, :, s : s + block].float() @ key.transpose(-1, -2)) * scaling  # [B, H, b, Tk]
            allowed = (kpos[None, :] <= qpos[:, None]).expand(B, 1, -1, -1).clone()
            if self.mode == "suppress":
                for r, span in enumerate(self.spans):
                    if span:
                        a, b = span
                        allowed[r, 0, qpos >= b, a:b] = False
            probs = torch.softmax(logits.masked_fill(~allowed, float("-inf")), dim=-1)
            out[:, :, s : s + block] = (probs.to(value.dtype) @ value).to(out.dtype)
            if self.mode == "receiver":
                to_seg = torch.zeros(H, len(qpos), self.n_seg, device=query.device).index_add_(2, self.seg, probs[0])
                self.sums[module.layer_idx].index_add_(1, self.seg[qpos], to_seg)
        return out.transpose(1, 2).contiguous(), None


@torch.no_grad()
def receiver(model, patch, ids, segs):
    """[layers * heads][segments] receiver score, None where no model-written segment is MIN_GAP or more later."""
    T, n = len(ids), len(segs)
    seg = torch.empty(T, dtype=torch.long)
    for k, sg in enumerate(segs):
        seg[sg["tok_start"] : sg["tok_end"]] = k
    patch.seg, patch.n_seg = seg.to(model.device), n
    H = model.config.num_attention_heads
    patch.sums = {L: torch.zeros(H, n, n, device=model.device) for L in patch.attn_layers}
    patch.mode = "receiver"
    try:
        model.model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
    finally:
        patch.mode = None
    size = torch.tensor([sg["tok_end"] - sg["tok_start"] for sg in segs], device=model.device, dtype=torch.float)
    gen = torch.tensor([sg["generated"] for sg in segs], device=model.device)
    idx = torch.arange(n, device=model.device)
    later = gen[:, None] & (idx[:, None] - idx[None, :] >= MIN_GAP)  # later[j, i]: j counts toward i's score
    count = later.sum(0)
    rows = []
    for L in patch.attn_layers:
        A = patch.sums[L] / (size[:, None] * size[None, :])  # [H, j, i] mean token attention
        v = (A * later).sum(1) / count.clamp(min=1)
        rows += [[round(float(x), 6) if c else None for x, c in zip(h, count)] for h in v.tolist()]
    return rows


@torch.no_grad()
def suppress(model, patch, ids, segs, batch_tokens):
    """kl[s][u], mean KL(normal || s suppressed) over u's model-written tokens (the prediction of each), u after s."""
    n = len(segs)
    owner = {t: k for k, sg in enumerate(segs) if sg["generated"] for t in range(sg["tok_start"], sg["tok_end"])}
    targets = sorted(t - 1 for t in owner if t >= 1)  # position t-1 predicts generated token t
    tgt_seg = torch.tensor([owner[t + 1] for t in targets], device=model.device)
    tpos = torch.tensor(targets, device=model.device)
    sources = [k for k, sg in enumerate(segs) if sg["field"] is not None and targets and sg["tok_end"] <= targets[-1]]
    kl = [[None] * n for _ in range(n)]
    B = max(2, min(16, batch_tokens // len(ids)))  # row 0 is the normal forward
    for i in range(0, len(sources), B - 1):
        chunk = sources[i : i + B - 1]
        patch.spans = [None] + [(segs[k]["tok_start"], segs[k]["tok_end"]) for k in chunk]
        patch.mode = "suppress"
        try:
            h = model.model(input_ids=torch.tensor([ids], device=model.device).expand(len(patch.spans), -1), use_cache=False)
            h = h.last_hidden_state[:, tpos]
        finally:
            patch.mode = None
        per_tok = torch.empty(len(chunk), len(targets), device=model.device)
        for s in range(0, len(targets), 256):
            logp = F.log_softmax(model.lm_head(h[:, s : s + 256]).float(), dim=-1)
            per_tok[:, s : s + 256] = (logp[0].exp() * (logp[0] - logp[1:])).sum(-1)
        for r, k in enumerate(chunk):
            after = tpos >= segs[k]["tok_end"]  # queries after the span: their view of it is what was removed
            tot = torch.zeros(n, device=model.device).index_add_(0, tgt_seg[after], per_tok[r, after].clamp(min=0))
            cnt = torch.zeros(n, device=model.device).index_add_(0, tgt_seg[after], torch.ones_like(per_tok[r, after]))
            for u in cnt.nonzero().flatten().tolist():
                kl[k][u] = round(float(tot[u] / cnt[u]), 6)
    return kl


@torch.no_grad()
def check(model, patch, ids, segs):
    """The patched attention matches sdpa, and with a mid-transcript segment suppressed, randomizing its tokens doesn't
    move any later logit."""
    x = torch.tensor([ids], device=model.device)
    patched = model.lm_head(model.model(input_ids=x, use_cache=False).last_hidden_state).float()
    ALL_ATTENTION_FUNCTIONS["sdpa"] = patch.orig
    try:
        plain = model.lm_head(model.model(input_ids=x, use_cache=False).last_hidden_state).float()
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = patch.attention
    agree = (patched.argmax(-1) == plain.argmax(-1)).float().mean().item()
    print(f"check: patched vs sdpa: max |dlogit| {(patched - plain).abs().max().item():.3f}, same argmax {agree:.4f}")
    k = next(k for k, sg in enumerate(segs) if sg["field"] is not None and sg["tok_end"] - sg["tok_start"] >= 3 and sg["tok_start"] > 20)
    a, b = segs[k]["tok_start"], segs[k]["tok_end"]
    x = torch.tensor([ids, ids], device=model.device)
    x[1, a:b] = torch.randint(1000, 100000, (b - a,), device=model.device)
    patch.spans, patch.mode = [(a, b), (a, b)], "suppress"
    try:
        h = model.model(input_ids=x, use_cache=False).last_hidden_state
    finally:
        patch.mode = None
    logits = model.lm_head(h[:, b:]).float()
    leak = (logits[0] - logits[1]).abs().max().item()
    effect = (logits[0] - patched[0, b:]).abs().max().item()
    print(f"check: segment {k} {segs[k]['text']!r}: max |dlogit| from randomizing it while suppressed {leak:.4f} "
          f"(suppressing it vs not: {effect:.3f})", flush=True)  # fmt: skip
    return leak, effect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("requests")
    ap.add_argument("out")
    ap.add_argument("--batch-tokens", type=int, default=65536, help="suppression rows per forward = this / transcript length")
    ap.add_argument("--check", action="store_true", help="leak test on the first transcript, then exit")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    rows = [json.loads(line) for line in Path(args.requests).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["id"] not in done][: args.limit or None]
    print(f"{len(rows)} transcripts to read", flush=True)

    tok, model = load_base()
    patch = Patch(model)
    t0 = time.time()
    for n, row in enumerate(rows):
        ta = time.time()
        msgs, ids, where = render(tok, row)
        segs = segments(tok, msgs, ids, where)
        if args.check:
            check(model, patch, ids, segs)
            return
        rec = {"id": row["id"], "n_tokens": len(ids), "layers": patch.attn_layers, "segments": segs}
        rec["vertical"] = receiver(model, patch, ids, segs)
        rec["kl"] = suppress(model, patch, ids, segs, args.batch_tokens)
        rec["seconds"] = round(time.time() - ta, 2)
        with out.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{n + 1}/{len(rows)}] {row['id']}: {len(ids)} tokens, {len(segs)} segments, {rec['seconds']}s "
              f"| {(n + 1) / (time.time() - t0) * 3600:.0f} transcripts/h", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
