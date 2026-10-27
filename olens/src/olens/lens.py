"""The agu18dec oracle-lens read contract: carrier prompt, marker slot, injection scale, bullet parsing, sampling.

Rebuilt from the checkpoint configs, the agu18dec README and wsbench.produce.methods.OLens; no `global_workspace`
code. The model side (capture, decode, AR) lives in olens.one_engine on vLLM; the HF transformers path (hook-LoRA,
HF generate, HF AR, the ceselder L42 lens, NLA and J++) was deleted 2026-10-27 (last at PR #229's first commit).
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE = "Qwen/Qwen3.6-27B"
ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints"


def blocks(model):
    return model.model.layers


@dataclass
class Sampling:
    temperature: float = 0.0  # 0 = greedy
    top_p: float = 0.95
    top_k: int = 64
    max_new_tokens: int = 160
    max_bullets: int | None = None  # stop a row once this many bullet lines are complete


GREEDY = Sampling()  # default for read(); never mutated

BULLET = re.compile(r"^\s*[-*•]", re.MULTILINE)


def bullets(text):
    return [m.strip() for m in re.findall(r"^\s*[-*•]\s+(.+)$", text, re.MULTILINE)]


def trim_bullets(text, k):
    """Keep the first k bullet lines (plus anything before the first bullet)."""
    starts = [m.start() for m in BULLET.finditer(text)]
    if len(starts) <= k:
        return text
    return text[: starts[k]].rstrip()


# The agu18dec lens to use: olens_s3d_space_rl600, not olens_s3d_rl600. The older one's RL reward stripped each
# bullet's leading space before the AR scored it, so it learned to open bullets with a Chinese word (36% of bullets
# contain Chinese vs 3%, same WorkspaceBench score; agu18dec/olens_and_ar README).
SPACE_LENS = CKPT / "agu18dec_olens_and_ar/olens_s3d_space_rl600"


def bullet_prefix(adapter_dir):
    """What goes before a bullet when the AR scores it: " " for the space lens (it was trained on " " + bullet),
    nothing for the old olens_s3d_rl600 (its reward stripped the space)."""
    return " " if "space" in Path(adapter_dir).name else ""


class AguLens:
    """agu18dec/olens_and_ar:olens_s3d_space_rl600 (or the older olens_s3d_rl600). Contract: prompt `concepts_raw`, the
    marker's INPUT EMBEDDING is overwritten with alpha * h/||h||, alpha = 16000, layers 20..60 step 4. Subclasses
    (olens.one_engine.OneEngine) set `self.tok` and `self._carriers = {}` and do the decoding."""

    layers = tuple(range(20, 61, 4))
    alpha = 16000.0
    marker_id = 158983  # ㈜ U+321C, verified per prompt in _carrier
    tok: Any
    _carriers: dict

    def prompt(self, layer):
        return (
            f"An activation vector from layer {layer} of a language model is enclosed in "
            f"activation tags: <activation>㈜</activation>. Produce distinct concepts "
            f"that encode this activation, each as a '- ' bullet on its own line."
        )

    def _carrier(self, layer):
        """Chat-rendered carrier prompt for `layer` -> (token ids, index of the marker slot), memoised."""
        if layer not in self._carriers:
            out = self.tok.apply_chat_template(
                [{"role": "user", "content": self.prompt(layer)}], tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            ids = list(out["input_ids"] if hasattr(out, "keys") else out)
            slots = [i for i, t in enumerate(ids) if t == self.marker_id]
            assert len(slots) == 1, f"marker survives as one token exactly once, got {slots}"
            self._carriers[layer] = (ids, slots[0])
        return self._carriers[layer]
