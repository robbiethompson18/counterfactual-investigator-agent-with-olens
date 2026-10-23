"""The ceselder L42 NLA's AV (olens.extras.NLA) decoded by vLLM instead of HF generate.

vLLM serves a copy of the base with both of the AV's LoRAs (SFT, then the RL LoRA trained on top) merged in
(scripts/merge_lens.py --adapter av_sft_lora --adapter av_rl_lora_step400). The activation goes in the way
olens.lens.CeselderLens injects it, h <- h + |h| * v/|v| at the output of decoder block 1 at the marker, via
ceselder/vllm-metamodels (`SteeringVector(norm_match=True, scale=1)` on the full residual stream; it also lets decode
run as CUDA graphs). Needs vLLM 0.19.0 + vllm-metamodels, and VLLM_LENS_CUDA_GRAPHS=1 set before vLLM is imported.

Create it BEFORE loading any HF model on the same GPU, so vLLM's memory profiling sees the GPU it will share.
"""

from pathlib import Path

import torch
import yaml

from .extras import EXPL

INJECT_LAYER = 1


class VllmNLA:
    def __init__(self, tok, merged_dir, meta_path, gpu_memory_utilization=0.9, max_num_seqs=1024):
        from vllm import LLM

        meta = yaml.safe_load(Path(meta_path).read_text())
        assert meta["extraction"]["layer_index"] == 42
        t = meta["tokens"]
        prompt = meta["prompt_templates"]["actor"].format(injection_char=t["injection_char"])
        out = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )  # as olens.lens.Lens._carrier
        self.ids = list(out["input_ids"] if hasattr(out, "keys") else out)
        slots = [i for i, x in enumerate(self.ids) if x == int(t["injection_token_id"])]
        assert len(slots) == 1, f"marker survives as one token exactly once, got {slots}"
        self.slot = slots[0]
        assert self.ids[self.slot - 1] == int(t["injection_left_neighbor_id"]) and self.ids[self.slot + 1] == int(
            t["injection_right_neighbor_id"]
        ), "marker neighbours drifted"
        self.llm = LLM(
            model=str(merged_dir),
            dtype="bfloat16",
            max_model_len=1024,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,  # <= 1024: vLLM 0.19's GDN decode kernel dies above (vllm-metamodels README)
            enable_prefix_caching=False,
        )

    @torch.no_grad()
    def explain(self, vecs, max_new_tokens=220, temperature=0.0):
        """[B, d] raw L42 activations -> [B] explanation strings, parsed as olens.extras.NLA.explain parses them."""
        from vllm import SamplingParams
        from vllm_lens import SteeringVector

        vecs = torch.as_tensor(vecs).float().cpu()
        params = [
            SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                stop=["</explanation>"],
                include_stop_str_in_output=True,
                extra_args={
                    "apply_steering_vectors": [
                        SteeringVector(
                            activations=v.view(1, 1, -1),
                            layer_indices=[INJECT_LAYER],
                            position_indices=[self.slot],
                            norm_match=True,
                            scale=1.0,
                        )
                    ]
                },
            )
            for v in vecs
        ]
        outs = self.llm.generate([{"prompt_token_ids": self.ids}] * len(vecs), params, use_tqdm=False)
        texts = [o.outputs[0].text for o in outs]
        return [(m.group(1) if (m := EXPL.search(t)) else t).strip() for t in texts]
