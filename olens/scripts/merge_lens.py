"""Bake a lens LoRA into a copy of the Qwen3.6-27B checkpoint (W += scale * B @ A, in fp32),
keeping the original file layout and config, so vLLM (whose LoRA runtime is inert on this
model) can serve the lens as a plain model. Repeat --adapter to stack several (their deltas add,
as stacked HookLoRAs do), e.g. the ceselder NLA's SFT LoRA plus the RL LoRA trained on top of it.

  python scripts/merge_lens.py --adapter checkpoints/agu18dec_olens_and_ar/olens_s3d_rl600 \
      --out /data/merged/agu_rl600
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True, action="append")
ap.add_argument("--out", required=True)
ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
ap.add_argument("--revision", default="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9")
args = ap.parse_args()

src = Path(snapshot_download(args.base, revision=args.revision))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
lora, scales = {}, {}  # checkpoint key -> [(scale, A, B)] per adapter
for adir in map(Path, args.adapter):
    cfg = json.loads((adir / "adapter_config.json").read_text())
    scale = scales[str(adir)] = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    ab = {}
    for k, v in load_file(adir / "adapter_model.safetensors").items():
        m = re.match(r"base_model\.model\.(?:model\.)?(?:language_model\.)?layers\.(\d+)\.(?:_orig_mod\.)?(.+)\.lora_([AB])\.weight$", k)
        assert m, k
        ab.setdefault(f"model.language_model.layers.{m[1]}.{m[2]}.weight", {})[m[3]] = v.float()
    for k, d in ab.items():
        lora.setdefault(k, []).append((scale, d["A"], d["B"]))

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
            sd[k] = (sd[k].float() + sum(scale * (B @ A) for scale, A, B in lora[k])).to(sd[k].dtype)
            done += 1
    save_file(sd, out / shard, metadata={"format": "pt"})
    print(f"{shard}: merged so far {done}/{len(lora)}", flush=True)
assert done == len(lora)
(out / "MERGED_FROM.json").write_text(json.dumps({"adapters": scales, "n_modules": done}))
print("ok", out)
