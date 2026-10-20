"""LoRA adapters applied with forward hooks, without PEFT.

Several adapters can sit on one base model; at most one should be `active` at a time. With every
adapter inactive the model is exactly the base model, which is what activations are captured on.
"""

import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

# PEFT key: base_model.model[.model].layers.N.[_orig_mod.]<path>.lora_{A,B}.weight
KEY = re.compile(
    r"base_model\.model\.(?:model\.)?(?:language_model\.)?layers\.(\d+)\."
    r"(?:_orig_mod\.)?(.+)\.lora_([AB])\.weight$"
)


def lora_scale(cfg: dict) -> float:
    """alpha/r, or alpha/sqrt(r) for rsLoRA."""
    r, a = cfg["r"], cfg["lora_alpha"]
    return a / r**0.5 if cfg.get("use_rslora") else a / r


class HookLoRA:
    """y = W x + scale * B A x on every adapted Linear."""

    def __init__(self, model, adapter_dir, device="cuda"):
        adapter_dir = Path(adapter_dir)
        self.cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        self.scale = lora_scale(self.cfg)
        sd = load_file(adapter_dir / "adapter_model.safetensors")
        mods = dict(model.named_modules())
        pairs = {}
        for k, v in sd.items():
            m = KEY.match(k)
            if not m:
                raise KeyError(f"unrecognised adapter key {k}")
            name = f"model.layers.{m[1]}.{m[2]}"
            pairs.setdefault(name, {})[m[3]] = v.to(device, torch.bfloat16)
        missing = [n for n in pairs if n not in mods]
        if missing:
            raise KeyError(f"adapter targets {len(missing)} missing modules, e.g. {missing[:3]}")
        self.pairs = pairs
        self.active = False
        self.handles = [mods[n].register_forward_hook(self._hook(ab["A"], ab["B"])) for n, ab in pairs.items()]

    def _hook(self, A, B):
        def f(mod, inp, out):
            if not self.active:
                return out
            return out + self.scale * F.linear(F.linear(inp[0], A), B)

        return f

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *exc):
        self.active = False

    def remove(self):
        for h in self.handles:
            h.remove()
