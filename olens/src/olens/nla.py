"""The all-layer NLA for Qwen3.6-27B on the one engine (agu18dec/qwen3.6-27b-all-layer-nla-rl2000, Camila Blank's, 2026-10-28).

The AV (activation verbalizer) is one rank-128 rsLoRA on every projection of every block, trained with EasyNLA's Qwen3.6
fork and an O-lens-style layer tag: the `actor` prompt names the source layer, the activation is injected Karvonen-style
(h <- h + |h| v/|v| at the output of decoder block 1, at the ㈜ marker), and the AV writes an <explanation> of 2-3 text
snippets, ~185 tokens. It reads any layer 0..63. Same recipe, prompts and marker as the 11-layer
camilablank/qwen3.6-27b-multilayer-nla-rl600 (layers 20..60 step 4, what we used until 2026-10-29), with RL to step 2000;
its model card: level with or above RL600 on WorkspaceBench at every layer (any of the 11 ladder layers 0.778 vs 0.758).

Here the AV runs through vLLM's LoRA runtime on the same engine that captures the activations (olens.one_engine); the
injection is vllm-metamodels' SteeringVector (norm_match=True, scale=1 at layer 1). The critic (AR, a full 64-block model
plus a value head) does not run here: scripts/olens_read_requests.py --readout nla saves every read's activation beside
the reads and scripts/nla_score.py adds the FVE later, on its own pod (docs/nla-reader.md).

Celeste's L42 NLA (ceselder/qwen3.6-27b-nla-rl, RL step 600) runs on the same class: its AV stacked into one rank-192 LoRA
on base Qwen (olens/scripts/ceselder_nla_lora.py), its own nla_meta.yaml (same marker and injection, no layer tag; the
actor template ignores `layer`), `repo=CESELDER_REPO`, and reads at layer 42 only.
"""

import json
import re
from pathlib import Path

import torch
import yaml

from .lens import Sampling
from .one_engine import OneEngine

NLA_REPO = "agu18dec/qwen3.6-27b-all-layer-nla-rl2000"
CESELDER_REPO = "ceselder/qwen3.6-27b-nla-rl"  # L42 only; AV = av_base (warmstart merged) + av_rl_adapters/iter_000600
EXPL = re.compile(r"<explanation>(.*?)(?:</explanation>|$)", re.DOTALL)
INJECT_LAYER = 1  # EasyNLA: Karvonen norm-matched add at the output of decoder block 1
NLA_SAMPLING = Sampling(max_new_tokens=200)  # greedy; the RL rollouts used max_new_tokens 200
AV_LORA = "/workspace/lora_vllm/nla64_av"  # the av/ adapter with vLLM-renamed keys (runpod/nla_reads.sh setup)
META = "/workspace/olens_ckpt/nla64/nla_meta.yaml"


class NLAEngine(OneEngine):
    layers = tuple(range(64))

    def __init__(self, tok, base="Qwen/Qwen3.6-27B", av_lora=AV_LORA, meta_path=META, repo=NLA_REPO, **kw):
        self.meta = yaml.safe_load(Path(meta_path).read_text())
        rank = json.loads((Path(av_lora) / "adapter_config.json").read_text())["r"]
        t = self.meta["tokens"]
        self.marker_id = int(t["injection_token_id"])
        self.neighbours = (int(t["injection_left_neighbor_id"]), int(t["injection_right_neighbor_id"]))
        super().__init__(
            tok,
            base,
            lens_lora=av_lora,
            ar_lora=None,
            ar_root=None,
            max_lora_rank=next(r for r in (64, 128, 256, 320, 512) if r >= rank),
            **kw,
        )
        self.adapter = f"{repo}:av"  # olens.provenance
        self.adapters = {
            "av_lora": str(av_lora),
            "engine": "vllm-metamodels one engine",
            "injection": f"karvonen add at block {INJECT_LAYER}",
        }
        self.bullet_prefix = None

    def prompt(self, layer):
        return self.meta["prompt_templates"]["actor"].format(layer=layer, injection_char=self.meta["tokens"]["injection_char"])

    def _carrier(self, layer):
        ids, slot = super()._carrier(layer)
        assert (ids[slot - 1], ids[slot + 1]) == self.neighbours, f"marker neighbours drifted: {ids[slot - 1 : slot + 2]}"
        return ids, slot

    @torch.no_grad()
    def read_items(self, vecs, layers, sampling=NLA_SAMPLING, n=1, seed=0):
        """vecs [B, d] raw activations at layers [B] -> [B][n] explanation strings, one generate with the AV LoRA."""
        from vllm import SamplingParams
        from vllm_lens import SteeringVector

        greedy = sampling.temperature <= 0
        prompts, params = [], []
        seeds = seed if isinstance(seed, list) else [seed] * len(vecs)
        for v, L, s in zip(vecs, layers, seeds):
            ids, slot = self._carrier(L)
            prompts.append({"prompt_token_ids": ids})
            act = torch.as_tensor(v).float().cpu().view(1, 1, -1)
            sv = SteeringVector(activations=act, layer_indices=[INJECT_LAYER], position_indices=[slot], norm_match=True, scale=1.0)
            params.append(
                SamplingParams(
                    n=n,
                    max_tokens=sampling.max_new_tokens,
                    temperature=0.0 if greedy else sampling.temperature,
                    top_p=1.0 if greedy else sampling.top_p,
                    top_k=-1 if greedy else sampling.top_k,
                    seed=None if greedy else s,
                    stop=["</explanation>"],
                    include_stop_str_in_output=True,
                    extra_args={"apply_steering_vectors": [sv]},
                )
            )
        outs = self._gen(prompts, params, lora=self.lens_req)
        return [[explanation(c.text) for c in o.outputs] for o in outs]


def explanation(text):
    m = EXPL.search(text)
    return (m.group(1) if m else text).strip()
