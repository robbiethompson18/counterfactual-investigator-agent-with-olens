"""The agu18dec lens decoded by vLLM instead of HF generate (~20x faster per read on an H200).

It serves a copy of the checkpoint with the lens merged in (scripts/merge_lens.py). vLLM's LoRA runtime also works
once the adapter keys are renamed (scripts/lora_for_vllm.py), but decodes ~19% slower. The carrier prompt goes in as
prompt embeddings, embedded with the model's own table and with the marker row overwritten by alpha * h/|h| - the
same contract as AguLens. Gate-set joint FVE: 0.4862 here vs 0.4859 for the HF path; greedy texts differ only
by bf16 kernel noise (same first bullet on ~80% of rows).

Create it BEFORE loading the HF model, so vLLM's memory profiling sees the GPU it will share.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .lens import GREEDY, AguLens, bullet_prefix, trim_bullets


class VllmLens(AguLens):
    def __init__(self, tok, merged_dir, gpu_memory_utilization=0.5, max_model_len=1024, max_num_seqs=128):
        from vllm import LLM

        self.tok, self._carriers, self.model = tok, {}, None  # model: the HF base, set later (Readout uses its device)
        d = Path(merged_dir)
        adapter = json.loads((d / "MERGED_FROM.json").read_text())["adapter"]
        self.adapter, self.merged_dir = Path(adapter).name, str(d)
        self.bullet_prefix = bullet_prefix(adapter)
        wmap = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
        key = "model.language_model.embed_tokens.weight"
        with safe_open(d / wmap[key], "pt") as f:
            self.E = f.get_tensor(key)  # [V, d] bf16, CPU
        self.llm = LLM(
            model=str(d),
            dtype="bfloat16",
            enable_prompt_embeds=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False,
            max_num_seqs=max_num_seqs,
        )

    def _embeds(self, layer, h):
        ids, slot = self._carrier(layer)
        e = self.E[torch.tensor(ids)].clone()
        h = torch.as_tensor(h).float().cpu()
        e[slot] = (self.alpha * h / h.norm().clamp_min(1e-9)).to(e.dtype)
        return {"prompt_embeds": e}

    def _params(self, sampling, n, seed):
        from vllm import SamplingParams

        greedy = sampling.temperature <= 0
        return SamplingParams(
            n=n,
            max_tokens=sampling.max_new_tokens,
            temperature=0.0 if greedy else sampling.temperature,
            top_p=1.0 if greedy else sampling.top_p,
            top_k=-1 if greedy else sampling.top_k,
            seed=None if greedy else seed,
        )

    def _finish(self, text, sampling):
        text = text.strip()
        return trim_bullets(text, sampling.max_bullets) if sampling.max_bullets else text

    @torch.no_grad()
    def read_layers(self, vecs, sampling=GREEDY, n=1, seed=0):
        layers = list(vecs)
        outs = self.llm.generate([self._embeds(L, vecs[L]) for L in layers], self._params(sampling, n, seed), use_tqdm=False)
        return {L: [self._finish(c.text, sampling) for c in o.outputs] for L, o in zip(layers, outs)}

    @torch.no_grad()
    def read_items(self, vecs, layers, sampling=GREEDY, n=1, seed=0):
        """vecs [B, d] at layers [B] -> [B][n] strings, all in one generate (vLLM batches them itself)."""
        outs = self.llm.generate([self._embeds(L, v) for v, L in zip(vecs, layers)], self._params(sampling, n, seed), use_tqdm=False)
        return [[self._finish(c.text, sampling) for c in o.outputs] for o in outs]

    def read(self, vecs, layer, sampling=GREEDY, n=1, seed=0):
        vecs = torch.as_tensor(vecs)
        if vecs.ndim == 1:
            vecs = vecs[None]
        outs = self.llm.generate([self._embeds(layer, v) for v in vecs], self._params(sampling, n, seed), use_tqdm=False)
        return [[self._finish(c.text, sampling) for c in o.outputs] for o in outs]
