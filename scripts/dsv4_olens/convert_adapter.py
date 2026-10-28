"""Rename Camila's V4 O-lens PEFT adapter (HF transformers module names) to the original checkpoint names vLLM's
DeepseekV4ForCausalLM maps from (its hf_to_vllm_mapper, packed fused_wqa_wkv / gate_up_proj).

    python convert_adapter.py /data/v4lens/lora /data/v4lens/lora_vllm
"""

import json
import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

NAMES = {
    "self_attn.q_a_proj": "attn.wq_a",
    "self_attn.q_b_proj": "attn.wq_b",
    "self_attn.kv_proj": "attn.wkv",
    "self_attn.o_b_proj": "attn.wo_b",
    "mlp.shared_experts.gate_proj": "ffn.shared_experts.w1",
    "mlp.shared_experts.up_proj": "ffn.shared_experts.w3",
    "mlp.shared_experts.down_proj": "ffn.shared_experts.w2",
}

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
w = load_file(str(src / "adapter_model.safetensors"))
out = {}
for k, v in w.items():
    m = re.match(r"base_model\.model\.model\.layers\.(\d+)\.(.+)\.(lora_[AB]\.weight)$", k)
    assert m and m.group(2) in NAMES, k
    out[f"base_model.model.layers.{m.group(1)}.{NAMES[m.group(2)]}.{m.group(3)}"] = v.contiguous()
save_file(out, str(dst / "adapter_model.safetensors"))
cfg = json.loads((src / "adapter_config.json").read_text())
cfg["target_modules"] = sorted({n.rsplit(".", 1)[1] for n in NAMES.values()})
cfg.pop("base_model_name_or_path", None)
(dst / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
print(f"{len(out)} tensors -> {dst}; target_modules {cfg['target_modules']}")
