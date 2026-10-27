"""One vLLM engine for all three reader jobs: capture of the transcript's residuals, the lens decode, and the AR.

Replaced the HF copy of Qwen (capture + AR, ~71% of the old reader's time at ~18% MFU) that shared the GPU with a vLLM
copy of the merged lens (deleted 2026-10-27; docs/olens-one-engine-reader.md). Needs vLLM 0.27.1 + ceselder/vllm-metamodels
(docs/metamodel-serving.md), whose hooks return the residual stream at requested prompt positions
(`extra_args={"output_residual_stream": [L], "capture_positions": [...]}`), and vLLM's LoRA runtime with the adapters'
keys renamed for the ConditionalGeneration module tree (olens/scripts/lora_for_vllm.py). Lens requests inject the
activation through `prompt_embeds` (the lens overwrites one input embedding, so no hook is needed for it).

Jobs, all on the base checkpoint:
  capture(ids, layers, positions)  no LoRA, max_tokens=1, residual after block L at the stop tokens
  read_items(vecs, layers)         lens LoRA, prompt_embeds with alpha*h/|h| at the marker; the lens ends its own list at 4 bullets
  ar(spans, layers)                AR LoRA, `[Layer NN] <span>`, residual after the last block at the last token, then the
                                   final RMSNorm (HF's last_hidden_state), the AR's layer norm and head
"""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

from .lens import CKPT, GREEDY, AguLens, bullet_prefix, trim_bullets

AR_ROOT = CKPT / "agu18dec_olens_and_ar/ar_ptag_pooled"
N_BLOCKS = 64  # Qwen3.6-27B decoder blocks; the AR reads the residual after the last one


class OneEngine(AguLens):
    def __init__(
        self,
        tok,
        base="Qwen/Qwen3.6-27B",
        lens_lora: str = "/workspace/lora_vllm/olens_space_rl600",
        ar_lora: str = "/workspace/lora_vllm/ar_ptag_pooled",
        ar_root=AR_ROOT,
        gpu_memory_utilization=0.88,
        max_num_seqs=512,
        max_model_len=32768,
        cuda_graphs=True,
        lora_mode="runtime",
        debug=False,
    ):
        """lora_mode: "runtime" applies the adapters through vLLM's LoRA kernels (every decode step pays for them);
        "merge" folds each adapter into the served weights with vllm-metamodels' merge_lora before its stage and unmerges
        after (a plain engine: no LoRA kernels in the CUDA graphs; ~1 s per publish with the base copy in pinned host
        memory, so stages are batched in olens_read_requests to amortise it)."""
        # vllm-metamodels turns torch.compile off unless asked (its hooks become a custom op); both default to on here
        os.environ.setdefault("VLLM_LENS_CUDA_GRAPHS", "1" if cuda_graphs else "0")
        os.environ.setdefault("VLLM_LENS_COMPILE", "1" if cuda_graphs else "0")
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.tok, self._carriers, self.debug, self.merge = tok, {}, debug, lora_mode == "merge"
        assert lora_mode in ("runtime", "merge"), lora_mode
        d = Path(base) if Path(base).exists() else Path(snapshot_download(base))
        wmap = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
        ek = next(k for k in wmap if k.endswith("embed_tokens.weight"))
        nk = next(k for k in wmap if k.endswith("norm.weight") and ".layers." not in k)
        with safe_open(d / wmap[ek], "pt") as f:
            self.E = f.get_tensor(ek)  # [V, d] bf16, CPU
        with safe_open(d / wmap[nk], "pt") as f:
            self.norm_w = f.get_tensor(nk).float().cuda()
        cfg = json.loads((d / "config.json").read_text())
        self.eps = float((cfg.get("text_config") or cfg).get("rms_norm_eps", 1e-6))
        self.llm = LLM(
            model=str(d),
            dtype="bfloat16",
            enable_lora=not self.merge,
            max_lora_rank=16,
            max_loras=2,
            enable_prompt_embeds=True,
            max_model_len=max_model_len,
            max_num_batched_tokens=max(max_model_len, 16384),  # one prefill chunk per transcript: capture hooks see every position
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=False,
            # the default FlashInfer GDN prefill (CuTeDSL JIT) never returned on a 98k-token single-chunk prefill (12 min at
            # 60% GPU, 2026-10-27); triton does it in 11 s. Prefill is ~2% of the reader's time, so nothing is lost on short ones.
            gdn_prefill_backend="triton",
        )
        self.lens_req = LoRARequest("olens", 1, str(lens_lora))
        self.ar_req = LoRARequest("ar", 2, str(ar_lora))
        self.merged = None  # merge mode: which adapter dir the served weights hold now
        self.bullet_prefix = bullet_prefix(lens_lora)
        self.adapter = "olens_s3d_space_rl600" if "space" in Path(lens_lora).name else Path(lens_lora).name  # olens.provenance
        self.adapters = {"lens_lora": str(lens_lora), "ar_lora": str(ar_lora), "engine": "vllm-metamodels one engine"}
        h = torch.load(Path(ar_root) / "heads.pt", map_location="cuda")
        self.W = h["head"]["linear.weight"].float()
        self.b = h["head"]["linear.bias"].float()
        self.tag_ids = h["tag_ids"]
        self.ar_layers = [int(L) for L in h["layers"]]

    def _gen(self, prompts, params, lora=None):
        if self.merge:
            self._serve(lora.lora_path if lora else None)
            lora = None
        return self.llm.generate(prompts, params, lora_request=lora, use_tqdm=False)

    def _serve(self, adapter):
        """Merge mode: make the served weights base + `adapter` (None = the clean base). One publish replaces the last."""
        if adapter == self.merged:
            return
        from vllm_lens.metamodel import merge_lora, unmerge_lora

        t = time.time()
        info = unmerge_lora(self.llm) if adapter is None else merge_lora(self.llm, adapter, keep_base="auto")
        self.merged = adapter
        if self.debug:
            print(f"{'unmerge' if adapter is None else 'merge ' + Path(adapter).name}: {time.time() - t:.2f}s {info}", flush=True)

    @staticmethod
    def _acts(out, n_layers, positions):
        """vllm-metamodels' residual_stream for one request -> [n_layers, P, d] fp32 CPU, checking the positions."""
        act = torch.as_tensor(out.activations["residual_stream"])
        got = [int(p) for p in out.activations["positions"]]
        if act.dim() == 2:
            act = act[None]
        if act.shape[0] != n_layers and act.shape[1] == n_layers:
            act = act.transpose(0, 1)
        assert act.shape[0] == n_layers and act.shape[1] == len(got), (tuple(act.shape), n_layers, len(got))
        want = [p if p >= 0 else len(out.prompt_token_ids) + p for p in positions]
        if got != want:
            order = {p: i for i, p in enumerate(got)}
            act = act[:, [order[p] for p in want]]
        return act.float().cpu()

    @torch.no_grad()
    def capture(self, ids, layers, positions):
        """Residual after block L at `positions` for each L -> {L: [P, d] fp32 CPU} (the HF capture()'s contract)."""
        layers = list(layers)
        if not positions:
            return {L: torch.empty(0, self.E.shape[1]) for L in layers}
        from vllm import SamplingParams

        p = SamplingParams(
            max_tokens=1, temperature=0.0, extra_args={"output_residual_stream": sorted(layers), "capture_positions": list(positions)}
        )
        out = self._gen([{"prompt_token_ids": list(ids)}], p)[0]
        act = self._acts(out, len(layers), list(positions))
        if self.debug:
            print(
                f"capture: {len(ids)} tokens, {len(positions)} positions, act {tuple(act.shape)}, "
                f"|h| L{min(layers)} = {act[0].norm(dim=-1).mean():.1f}",
                flush=True,
            )
        return {L: act[i] for i, L in enumerate(sorted(layers))}

    @torch.no_grad()
    def capture_many(self, reqs):
        """[(ids, layers, positions)] -> [{L: [P, d]}], every transcript's prefill in one generate."""
        from vllm import SamplingParams

        d = self.E.shape[1]
        res = [{L: torch.empty(0, d) for L in layers} for ids, layers, pos in reqs]
        idx = [i for i, (ids, layers, pos) in enumerate(reqs) if pos]
        if not idx:
            return res
        prompts = [{"prompt_token_ids": list(reqs[i][0])} for i in idx]
        params = [
            SamplingParams(
                max_tokens=1,
                temperature=0.0,
                extra_args={"output_residual_stream": sorted(reqs[i][1]), "capture_positions": list(reqs[i][2])},
            )
            for i in idx
        ]
        for i, out in zip(idx, self._gen(prompts, params)):
            _, layers, pos = reqs[i]
            act = self._acts(out, len(layers), list(pos))
            res[i] = {L: act[j] for j, L in enumerate(sorted(layers))}
        if self.debug:
            print(
                f"capture_many: {len(idx)} transcripts, {sum(len(r[0]) for r in reqs)} tokens, {sum(len(r[2]) for r in reqs)} positions",
                flush=True,
            )
        return res

    def _embeds(self, layer, h):
        ids, slot = self._carrier(layer)
        e = self.E[torch.tensor(ids)].clone()
        h = torch.as_tensor(h).float().cpu()
        e[slot] = (self.alpha * h / h.norm().clamp_min(1e-9)).to(e.dtype)
        return {"prompt_embeds": e}

    @torch.no_grad()
    def read_items(self, vecs, layers, sampling=GREEDY, n=1, seed=0):
        """vecs [B, d] at layers [B] -> [B][n] strings, one generate with the lens LoRA, trimmed to sampling.max_bullets. The space
        lens stops itself at 4 bullets 97% of the time and never reaches the 160-token cap, so no stop rule runs in the engine
        (a logits processor that forced <|im_end|> at the 5th bullet cost 6% per step for nothing; docs/olens-one-engine-reader.md)."""
        from vllm import SamplingParams

        greedy = sampling.temperature <= 0
        params = SamplingParams(
            n=n,
            max_tokens=sampling.max_new_tokens,
            temperature=0.0 if greedy else sampling.temperature,
            top_p=1.0 if greedy else sampling.top_p,
            top_k=-1 if greedy else sampling.top_k,
            seed=None if greedy else seed,
        )
        outs = self._gen([self._embeds(L, v) for v, L in zip(vecs, layers)], params, lora=self.lens_req)
        return [[self._finish(c.text, sampling) for c in o.outputs] for o in outs]

    def _finish(self, text, sampling):
        text = text.strip()
        return trim_bullets(text, sampling.max_bullets) if sampling.max_bullets else text

    @torch.no_grad()
    def ar(self, spans, layer):
        """spans: token-id lists; layer: one or one per span -> [B, d] AR head outputs (olens.ar.AR's contract)."""
        from vllm import SamplingParams

        layers = layer if isinstance(layer, (list, tuple)) else [layer] * len(spans)
        prompts = [{"prompt_token_ids": self.tag_ids[self.ar_layers.index(L)].tolist() + list(s)} for s, L in zip(spans, layers)]
        p = SamplingParams(max_tokens=1, temperature=0.0, extra_args={"output_residual_stream": [N_BLOCKS - 1], "capture_positions": [-1]})
        outs = self._gen(prompts, p, lora=self.ar_req)
        h = torch.stack([self._acts(o, 1, [-1])[0, -1] for o in outs]).cuda()
        # HF's last_hidden_state is the final RMSNorm of this residual: w * x * rsqrt(mean(x^2) + eps)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm_w
        return F.layer_norm(h, h.shape[-1:]) @ self.W.T + self.b
