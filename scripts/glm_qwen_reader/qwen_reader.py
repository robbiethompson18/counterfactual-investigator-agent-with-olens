"""Read saved GLM-5.3 block-51 vectors with ceselder/glm-5.3-olens-qwen-reader: v = A(h51) = h51 @ W.T + b (affine.pt, 6144 -> 5120),
added norm-matched to Qwen3.6-27B's block-1 output at the marker token of the public Qwen L42 lens prompt (its olens_usage.py:
LENS_MSG, _prompt, BulletGrammar), then the reader LoRA decodes 4 bullets under the grammar: greedy + --samples draws at T=1.
HF transformers + peft on one GPU, as the model card's usage; shard over GPUs with --shard i/n.

    CUDA_VISIBLE_DEVICES=0 python qwen_reader.py H51.pt READER_DIR L42_USAGE.py OUT.jsonl --shard 0/4

OUT.jsonl: {id, pos, reads: [greedy, s1, ...]} per vector (each read the raw "* a\\n* b\\n..." text).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

BASE, BASE_REV = "Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

ap = argparse.ArgumentParser()
ap.add_argument("h51")
ap.add_argument("reader", help="a checkpoint folder of ceselder/glm-5.3-olens-qwen-reader (release/)")
ap.add_argument("usage", help="olens_usage.py of ceselder/olens-qwen3.6-27b-l42")
ap.add_argument("out")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--batch", type=int, default=128, help="vectors per generate (samples multiply it)")
ap.add_argument("--max-new", type=int, default=80)
a = ap.parse_args()

sys.path.insert(0, str(Path(a.usage).parent))
import olens_usage as L42

k, n = map(int, a.shard.split("/"))
H = torch.load(a.h51)
idx = list(range(k, len(H["pos"]), n))
A = torch.load(f"{a.reader}/affine.pt")
V = (H["h51"][idx].cuda() @ A["W"].cuda().T + A["b"].cuda()).float()

tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
model = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REV, dtype=torch.bfloat16, device_map={"": 0})
model = PeftModel.from_pretrained(model, a.reader).eval()
layers = next(m for name, m in model.named_modules() if name.endswith("layers") and isinstance(m, torch.nn.ModuleList) and len(m) == 64)
ids, mpos = L42._prompt(tok, L42.LENS_MSG, "cuda")
eos = tok.convert_tokens_to_ids("<|im_end|>")
cur = {}


def hook(_m, _i, out):
    h = out[0] if isinstance(out, tuple) else out
    if h.shape[1] <= mpos:  # decode steps
        return out
    h = h.clone()
    base = h[:, mpos]
    h[:, mpos] = base + base.norm(dim=-1, keepdim=True) * F.normalize(cur["v"].to(h.dtype), dim=-1)
    return (h, *out[1:]) if isinstance(out, tuple) else h


layers[L42.INJECT_LAYER].register_forward_hook(hook)
grammar = L42.BulletGrammar(tok, ids.shape[1], model.get_output_embeddings().weight.shape[0], "cuda")


@torch.no_grad()
def gen(v, sample):
    cur["v"] = v
    x = ids.expand(v.shape[0], -1)
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=a.max_new, do_sample=sample,
                         temperature=1.0 if sample else None, top_p=1.0, top_k=0, logits_processor=LogitsProcessorList([grammar]),
                         eos_token_id=eos, pad_token_id=eos)  # fmt: skip
    return [tok.decode(o[ids.shape[1] :], skip_special_tokens=True).strip() for o in out]


with open(a.out, "w") as f:
    for c in range(0, len(idx), a.batch):
        v = V[c : c + a.batch]
        reads = [[g] for g in gen(v, False)]
        if a.samples:
            s = gen(v.repeat_interleave(a.samples, 0), True)
            for i, r in enumerate(reads):
                r += s[i * a.samples : (i + 1) * a.samples]
        for j, r in zip(idx[c : c + a.batch], reads):
            f.write(json.dumps({"id": H["ids"][j], "pos": H["pos"][j], "reads": r}, ensure_ascii=False) + "\n")
        f.flush()
        print(f"shard {a.shard}: {min(c + a.batch, len(idx))}/{len(idx)}", flush=True)
