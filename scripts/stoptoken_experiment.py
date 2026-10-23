"""More stop tokens, then O-lens reads at them: a qualitative side experiment on WeirdChat prompts.

Stop tokens = every vocab token matching the viewer's PUNCT regex (Kylie's read positions,
transcript-viewer/web/server.py). Base Qwen3.6-27B, thinking off, T=1 with top-k 20 / top-p 0.95
applied after the arm's edit. Arms, N rollouts per prompt each:

- control:  plain sampling
- up5:      stop tokens' probabilities x5 (logit + ln 5) before truncation
- force10:  if p(stop set) > 10% under the raw distribution, sample only from the stop set
- insert:   no new rollouts; on the control rollouts append "." after a word every ~12 tokens and
            read at that appended token (then discard it)

Reads (agu18dec lens + AR via transcript-viewer/pod/server.py, greedy, 4 bullets) at every stop
token of the response and its final <|im_end|>, at LAYERS. With --everywhere, rollout 0 of each
arm of the first prompt is also read at every response token (insert: "." after every token) at
layer 44.

    PYTHONPATH=olens/src:transcript-viewer/pod python scripts/stoptoken_experiment.py drunk_driving,human_alex [--everywhere]
writes data/runs/stoptokens/{rollouts,reads}_<first prompt>.jsonl
"""

import json
import math
import re
import sys
import time
from pathlib import Path

import server  # transcript-viewer/pod/server.py: read_many/score over its module-level state S
import torch
from olens.lens import Sampling
from transformers import LogitsProcessor

PROMPTS = json.loads(Path("data/runs/stoptokens/prompts.json").read_text())
KEYS = sys.argv[1].split(",")
EVERYWHERE = "--everywhere" in sys.argv
OUT = Path("data/runs/stoptokens")
ARMS = ("control", "up5", "force10")
N, MAX_NEW = 16, 400
LAYERS = (44, 56)
EVERY_LAYER = 44
INSERT_EVERY = 12
TOP_K, TOP_P, UP, THRESH = 20, 0.95, math.log(5), 0.10
PUNCT = re.compile(r"^\s*[.,;:!?]+[\"')\]*]*\s*$")  # = transcript-viewer/web/server.py PUNCT


class StopBias(LogitsProcessor):
    """Per-row arm edit, then top-k/top-p. Logs p(stop set) under the raw distribution per step."""

    def __init__(self, stop, arms):
        self.stop, self.arms, self.p_stop = stop, arms, []

    def __call__(self, input_ids, scores):
        s = scores.float()
        stop = self.stop[: s.shape[-1]]
        p = s.softmax(-1)[:, stop].sum(-1)
        self.p_stop.append(p.cpu())
        s = s + (self.arms == 1)[:, None] * stop[None] * UP
        force = ((self.arms == 2) & (p > THRESH))[:, None]
        s = s.masked_fill(force & ~stop[None], -math.inf)
        kth = s.topk(TOP_K, -1).values[:, -1:]
        s = s.masked_fill(s < kth, -math.inf)
        srt, idx = s.sort(-1, descending=True)
        drop = srt.softmax(-1).cumsum(-1) - srt.softmax(-1) > TOP_P
        return s.masked_fill(torch.zeros_like(drop).scatter(-1, idx, drop), -math.inf)


def stop_mask(tok, size):
    m = torch.zeros(size, dtype=torch.bool)
    special = set(tok.all_special_ids)
    for i in range(len(tok)):
        if i not in special and PUNCT.match(tok.decode([i])):
            m[i] = True
    return m


@torch.no_grad()
def rollouts(tok, model):
    end = tok.convert_tokens_to_ids("<|im_end|>")
    prompts = {
        k: tok.apply_chat_template(PROMPTS[k]["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for k in KEYS
    }
    rows = [(k, arm, s) for k in KEYS for arm in ARMS for s in range(N)]
    stop = stop_mask(tok, model.get_output_embeddings().weight.shape[0]).to(model.device)
    print(f"{int(stop.sum())} stop tokens", flush=True)
    proc = StopBias(stop, torch.tensor([ARMS.index(a) for _, a, _ in rows], device=model.device))
    enc = tok([prompts[k] for k, _, _ in rows], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    torch.manual_seed(0)
    gen = model.generate(
        **enc,
        max_new_tokens=MAX_NEW,
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        logits_processor=[proc],
        pad_token_id=tok.pad_token_id,
    )
    new, p_stop = gen[:, enc.input_ids.shape[1] :].tolist(), torch.stack(proc.p_stop, 1).tolist()
    out = []
    for (k, arm, s), ids, ps in zip(rows, new, p_stop, strict=True):
        n = ids.index(end) + 1 if end in ids else len(ids)
        ids, ps = ids[:n], ps[:n]
        prompt = tok(prompts[k], add_special_tokens=False).input_ids
        toks = [tok.decode([t]) for t in ids]
        out.append(
            {
                "key": k,
                "arm": arm,
                "sample": s,
                "prompt_ids": prompt,
                "ids": ids,
                "tokens": toks,
                "p_stop": [round(p, 3) for p in ps],
                "forced": [arm == "force10" and p > THRESH for p in ps],
                "is_stop": [bool(PUNCT.match(t)) for t in toks],
                "text": tok.decode(ids, skip_special_tokens=True),
                "truncated": end not in ids,
            }
        )
    return out


def read_jobs(tok, rolls):
    """(priority, rollout index, kind, response pos, layer, ids to capture or None = the rollout's own)."""
    dot = tok(".", add_special_tokens=False).input_ids
    assert len(dot) == 1
    jobs = []
    for ri, r in enumerate(rolls):
        stops = [i for i, t in enumerate(r["is_stop"]) if t] + ([len(r["ids"]) - 1] if not r["truncated"] else [])
        for li, L in enumerate(LAYERS):
            jobs += [(2 * li, ri, "stop", i, L, None) for i in stops]
        if r["arm"] == "control":
            last = -INSERT_EVERY
            for i in range(len(r["ids"]) - 1):
                nxt = r["tokens"][i + 1]
                if not r["is_stop"][i] and nxt[:1].isspace() and i - last >= INSERT_EVERY:
                    last = i
                    for li, L in enumerate(LAYERS):
                        jobs.append((2 * li + 1, ri, "insert", i, L, r["prompt_ids"] + r["ids"][: i + 1] + dot))
        if EVERYWHERE and r["key"] == KEYS[0] and r["sample"] == 0:
            for i in range(len(r["ids"])):
                jobs.append((1, ri, "every", i, EVERY_LAYER, None))
                if r["arm"] == "control":
                    jobs.append((1, ri, "every_insert", i, EVERY_LAYER, r["prompt_ids"] + r["ids"][: i + 1] + dot))
    return sorted(jobs, key=lambda j: j[0])


@torch.no_grad()
def capture_many(model, seqs, keep, bs=16, bucket=64):
    """Residual stream at LAYERS for many sequences: [{L: a[keep[i]]}]. Right-padded to a multiple of
    `bucket` and batched; causal, so the padding never reaches the kept positions, and the
    linear-attention kernels only see a few distinct lengths (they compile per length)."""
    out = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for b in range(0, len(order), bs):
        idx = order[b : b + bs]
        T = -(-max(len(seqs[i]) for i in idx) // bucket) * bucket
        ids = torch.tensor([seqs[i] + [0] * (T - len(seqs[i])) for i in idx], device=model.device)
        hs = model.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        for j, i in enumerate(idx):
            out[i] = {L: hs[L + 1][j, keep[i]].float().cpu() for L in LAYERS}
        del hs
    return out


def vectors(model, rolls, jobs):
    """One capture per rollout for its own positions; one per inserted-probe sequence."""
    own = capture_many(model, [r["prompt_ids"] + r["ids"] for r in rolls], [slice(len(r["prompt_ids"]), None) for r in rolls])
    keys = sorted({(ri, i) for _, ri, _, i, _, ids in jobs if ids is not None})
    seqs = {(ri, i): ids for _, ri, _, i, _, ids in jobs if ids is not None}
    probe = dict(zip(keys, capture_many(model, [seqs[k] for k in keys], [len(seqs[k]) - 1 for k in keys]), strict=True))
    return torch.stack([own[ri][L][i] if ids is None else probe[(ri, i)][L] for _, ri, _, i, L, ids in jobs])


def main():
    t0 = time.time()
    tok, model = server.load(batch=192)
    tok.padding_side = "left"
    print(f"loaded {time.time() - t0:.0f}s", flush=True)

    rolls = rollouts(tok, model)
    with (OUT / f"rollouts_{KEYS[0]}.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rolls)
    print(f"{len(rolls)} rollouts, {sum(len(r['ids']) for r in rolls)} tokens, {time.time() - t0:.0f}s", flush=True)

    jobs = read_jobs(tok, rolls)
    vecs = vectors(model, rolls, jobs)
    print(f"{len(jobs)} reads queued, vectors {time.time() - t0:.0f}s", flush=True)
    smp = Sampling(temperature=0, max_new_tokens=160, max_bullets=4)
    with (OUT / f"reads_{KEYS[0]}.jsonl").open("w") as f:
        for b in range(0, len(jobs), 960):
            chunk = jobs[b : b + 960]
            res = server.read_many(vecs[b : b + 960], [j[4] for j in chunk], smp)
            for (_, ri, kind, i, L, _), (rd,) in zip(chunk, res, strict=True):
                r = rolls[ri]
                f.write(
                    json.dumps(
                        {
                            "key": r["key"],
                            "arm": r["arm"],
                            "sample": r["sample"],
                            "kind": kind,
                            "pos": i,
                            "token": r["tokens"][i],
                            "layer": L,
                            **rd,
                        }
                    )
                    + "\n"
                )
            f.flush()
            print(f"read {b + len(chunk)}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
