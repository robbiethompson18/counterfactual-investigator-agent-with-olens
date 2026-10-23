"""Copy a PEFT LoRA trained on HF's text-only Qwen3.6-27B so vLLM's LoRA runtime applies it.

The adapters name modules `base_model.model.model.layers.N...` (HF's Qwen3_5ForCausalLM). vLLM loads the checkpoint
as Qwen3_5ForConditionalGeneration, whose weight mapper only rewrites `model.language_model.` prefixes, so no module
gets its weights. vLLM 0.32 raises nothing (its DEBUG log even says "Successfully loaded LoRA weights" for every
module) and serves the base model: Agam's "LoRA is inert". Measured on 792 greedy lens reads: with raw keys the text
matches base Qwen's exactly on 430 and agrees with the merged lens on the first token 1% of the time, as base does.
Renamed to
`model.language_model.layers.N...`, the LoRA matches the merged lens's first token on 792/792 and first line on 87%
(bf16 noise; HF vs vLLM-merged agree on ~80% of first bullets).

    python scripts/lora_for_vllm.py checkpoints/agu18dec_olens_and_ar/olens_s3d_space_rl600 /workspace/lora/olens_space_rl600
"""

import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
shutil.copy(src / "adapter_config.json", out / "adapter_config.json")
sd = load_file(src / "adapter_model.safetensors")
renamed = {k.replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers."): v for k, v in sd.items()}
assert all(".language_model.layers." in k for k in renamed), "unexpected key layout"
save_file(renamed, out / "adapter_model.safetensors")
print(f"{len(renamed)} tensors -> {out}")
