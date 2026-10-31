"""Celeste's L42 NLA AV (ceselder/qwen3.6-27b-nla-rl) as ONE vLLM LoRA on base Qwen3.6-27B, so olens.nla.NLAEngine serves it
on the capture engine like the all-layer NLA.

The repo ships `av_base/` (Qwen3.6-27B with the warmstart AV LoRA, r128 rsLoRA, merged in) and the GRPO LoRAs
`av_rl_adapters/iter_*` (r64 rsLoRA) trained on top of it. av_base is exactly base + `av_warmstart_lora/iter_0003864`: on
L23 o_proj and L41 in_proj_qkv, |av_base - (base + s B A)| / |av_base - base| = 0.020 / 0.015, max abs 0.0019 / 0.0005
(bf16 rounding of the merge; iter_0003000 leaves 0.10). So the AV = base + s1 B1 A1 + s2 B2 A2, which is one LoRA of rank
128 + 64 = 192 with A = [A1; A2], B = [s1 B1, s2 B2] and scale 1 (alpha = r, no rsLoRA). Keys renamed for vLLM as
lora_for_vllm.py does; load with max_lora_rank 256.

    python olens/scripts/ceselder_nla_lora.py iter_000600 /data/lora_vllm/ceselder_nla_rl600
"""

import json
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

REPO, WARM = "ceselder/qwen3.6-27b-nla-rl", "av_warmstart_lora/iter_0003864"
step, out = sys.argv[1], Path(sys.argv[2])


def adapter(sub):
    cfg = json.loads(Path(hf_hub_download(REPO, f"{sub}/adapter_config.json")).read_text())
    assert cfg["use_rslora"], cfg
    sd = load_file(hf_hub_download(REPO, f"{sub}/adapter_model.safetensors"))
    key = lambda k: re.sub(
        r"^base_model\.model\.(?:model\.)?(?:language_model\.)?layers\.", "base_model.model.model.language_model.layers.", k
    )
    return cfg, cfg["lora_alpha"] / cfg["r"] ** 0.5, {key(k): v.float() for k, v in sd.items()}


(c1, s1, w1), (c2, s2, w2) = adapter(WARM), adapter(f"av_rl_adapters/{step}")
assert w1.keys() == w2.keys(), set(w1) ^ set(w2)
sd = {}
for k in w1:
    if ".lora_A." in k:
        sd[k] = torch.cat([w1[k], w2[k]]).to(torch.bfloat16).contiguous()
    else:
        sd[k] = torch.cat([s1 * w1[k], s2 * w2[k]], 1).to(torch.bfloat16).contiguous()
r = c1["r"] + c2["r"]
cfg = c1 | {"r": r, "lora_alpha": r, "use_rslora": False, "base_model_name_or_path": "Qwen/Qwen3.6-27B",
            "target_modules": sorted(set(c1["target_modules"]) | set(c2["target_modules"]))}  # fmt: skip
out.mkdir(parents=True, exist_ok=True)
save_file(sd, out / "adapter_model.safetensors")
(out / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
(out / "stacked_from.json").write_text(json.dumps({"repo": REPO, "warmstart": WARM, "rl": step, "scales": [s1, s2]}))
print(f"{len(sd)} tensors, rank {r} -> {out}")
