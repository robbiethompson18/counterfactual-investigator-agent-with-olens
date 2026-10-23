"""Bake a lens LoRA into a copy of the Qwen3.6-27B checkpoint (W += scale * B @ A, in fp32),
keeping the original file layout and config, so vLLM can serve the lens as a plain model. (vLLM's LoRA runtime also
works once the keys are renamed, scripts/lora_for_vllm.py, but merged weights decode ~19% faster.)

  python scripts/merge_lens.py --adapter checkpoints/agu18dec_olens_and_ar/olens_s3d_space_rl600 \
      --out /data/merged/agu_space_rl600
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
ap.add_argument("--revision", default="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9")
args = ap.parse_args()

src = Path(snapshot_download(args.base, revision=args.revision))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
adir = Path(args.adapter)
cfg = json.loads((adir / "adapter_config.json").read_text())
scale = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])

lora = {}
for k, v in load_file(adir / "adapter_model.safetensors").items():
    m = re.match(r"base_model\.model\.(?:model\.)?(?:language_model\.)?layers\.(\d+)\.(?:_orig_mod\.)?(.+)\.lora_([AB])\.weight$", k)
    assert m, k
    lora.setdefault(f"model.language_model.layers.{m[1]}.{m[2]}.weight", {})[m[3]] = v.float()

wmap = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
missing = [k for k in lora if k not in wmap]
assert not missing, f"adapter targets not in checkpoint: {missing[:3]}"

for f in src.iterdir():  # configs, tokenizer, index: copy as-is
    if not f.name.endswith(".safetensors"):
        shutil.copy(f, out / f.name)

done = 0
for shard in sorted({wmap[k] for k in wmap}):
    sd = load_file(src / shard)
    for k in sd:
        if k in lora:
            ab = lora[k]
            sd[k] = (sd[k].float() + scale * (ab["B"] @ ab["A"])).to(sd[k].dtype)
            done += 1
    save_file(sd, out / shard, metadata={"format": "pt"})
    print(f"{shard}: merged so far {done}/{len(lora)}", flush=True)
assert done == len(lora)
(out / "MERGED_FROM.json").write_text(json.dumps({"adapter": str(adir), "scale": scale, "n_modules": done}))
print("ok", out)
