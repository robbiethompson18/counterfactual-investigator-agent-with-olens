"""Base model, activation capture, and the two oracle-lens read contracts.

Rebuilt from the checkpoint configs, the agu18dec README, wsbench.produce.methods.OLens and
EasyNLA's karvonen hook; no `global_workspace` code. See docs/summary.md.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .lora import HookLoRA

BASE = "Qwen/Qwen3.6-27B"
ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints"


def load_base(device="cuda"):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    tok = AutoTokenizer.from_pretrained(BASE)
    # explicit HF config class: importing vLLM re-registers "qwen3_5" in AutoConfig with its own
    # class, which AutoModelForCausalLM can't build from. No device_map: one device.
    model = AutoModelForCausalLM.from_pretrained(BASE, config=Qwen3_5Config.from_pretrained(BASE), dtype=torch.bfloat16).to(device)
    model.eval().requires_grad_(False)
    return tok, model


def blocks(model):
    return model.model.layers


@torch.no_grad()
def capture(model, ids, layers):
    """Residual stream after decoder block L (= hidden_states[L+1]) for each L: {L: [T, d] fp32}.
    Call with every lens inactive."""
    ids = torch.as_tensor(ids, device=model.device).view(1, -1)
    out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return {L: out.hidden_states[L + 1][0].float() for L in layers}


@dataclass
class Sampling:
    temperature: float = 0.0  # 0 = greedy
    top_p: float = 0.95
    top_k: int = 64
    max_new_tokens: int = 160
    max_bullets: int | None = None  # stop a row once this many bullet lines are complete
    min_new_tokens: int = 0

    def kwargs(self):
        kw = {"min_new_tokens": self.min_new_tokens} if self.min_new_tokens else {}
        if self.temperature <= 0:
            return dict(do_sample=False, **kw)
        return dict(do_sample=True, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, **kw)


GREEDY = Sampling()  # default for read(); never mutated

BULLET = re.compile(r"^\s*[-*•]", re.MULTILINE)


def trim_bullets(text, k):
    """Keep the first k bullet lines (plus anything before the first bullet)."""
    starts = [m.start() for m in BULLET.finditer(text)]
    if len(starts) <= k:
        return text
    return text[: starts[k]].rstrip()


class BulletStop(StoppingCriteria):
    """Per-row stop once the generated text holds k complete bullets, i.e. a (k+1)-th line
    starts. `skip` = prompt tokens at the front of input_ids (0 with inputs_embeds)."""

    def __init__(self, tok, k, skip=0):
        self.tok, self.k, self.skip = tok, k, skip

    def __call__(self, input_ids, scores, **kw):
        texts = self.tok.batch_decode(input_ids[:, self.skip :], skip_special_tokens=True)
        done = [len(BULLET.findall(t.strip())) > self.k for t in texts]
        return torch.tensor(done, device=input_ids.device)


class Lens:
    """Shared machinery: a hook-LoRA on the shared base, a chat-rendered carrier prompt with one
    marker slot, and batched generation. Subclasses define the prompt and the injection."""

    layers: tuple
    marker_id = 158983  # ㈜ U+321C, verified per prompt in _carrier

    def __init__(self, tok, model, adapter_dir):
        self.tok, self.model = tok, model
        self.lora = HookLoRA(model, adapter_dir, model.device)
        self._carriers = {}

    def prompt(self, layer) -> str:
        raise NotImplementedError

    def _carrier(self, layer):
        if layer not in self._carriers:
            out = self.tok.apply_chat_template(
                [{"role": "user", "content": self.prompt(layer)}], tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            ids = list(out["input_ids"] if hasattr(out, "keys") else out)
            slots = [i for i, t in enumerate(ids) if t == self.marker_id]
            assert len(slots) == 1, f"marker survives as one token exactly once, got {slots}"
            self._carriers[layer] = (ids, slots[0])
        return self._carriers[layer]

    @torch.no_grad()
    def read(self, vecs, layer, sampling=GREEDY, n=1, seed=0):
        """vecs: [B, d] (or [d]) residual activations captured from the base model.
        Returns [B][n] decoded strings. Subclasses' _generate returns only the new tokens."""
        vecs = torch.as_tensor(vecs).float().to(self.model.device)
        if vecs.ndim == 1:
            vecs = vecs[None]
        vecs = vecs.repeat_interleave(n, 0)
        ids, slot = self._carrier(layer)
        torch.manual_seed(seed)
        with self.lora:
            out = self._generate(ids, slot, vecs, sampling)
        texts = self.tok.batch_decode(out, skip_special_tokens=True)
        texts = [self._finish(t, sampling) for t in texts]
        return [texts[i * n : (i + 1) * n] for i in range(len(texts) // n)]

    def _finish(self, text, sampling):
        text = text.strip()
        return trim_bullets(text, sampling.max_bullets) if sampling.max_bullets else text

    def _gen_kwargs(self, sampling, batch, skip=0):
        pad = self.tok.pad_token_id or self.tok.eos_token_id
        kw = dict(
            max_new_tokens=sampling.max_new_tokens,
            pad_token_id=pad,
            attention_mask=torch.ones(batch, 1, dtype=torch.long),
            **sampling.kwargs(),
        )
        if sampling.max_bullets:
            kw["stopping_criteria"] = StoppingCriteriaList([BulletStop(self.tok, sampling.max_bullets, skip)])
        return kw


# The agu18dec lens to use: olens_s3d_space_rl600, not olens_s3d_rl600. The older one's RL reward stripped each
# bullet's leading space before the AR scored it, so it learned to open bullets with a Chinese word (36% of bullets
# contain Chinese vs 3%, same WorkspaceBench score; agu18dec/olens_and_ar README).
SPACE_LENS = CKPT / "agu18dec_olens_and_ar/olens_s3d_space_rl600"


def bullet_prefix(adapter_dir):
    """What goes before a bullet when the AR scores it: " " for the space lens (it was trained on " " + bullet),
    nothing for the old olens_s3d_rl600 (its reward stripped the space)."""
    return " " if "space" in Path(adapter_dir).name else ""


class AguLens(Lens):
    """agu18dec/olens_and_ar:olens_s3d_space_rl600 (or the older olens_s3d_rl600). Contract: prompt `concepts_raw`, the
    marker's INPUT EMBEDDING is overwritten with alpha * h/||h||, alpha = 16000, layers 20..60 step 4."""

    layers = tuple(range(20, 61, 4))
    alpha = 16000.0

    def __init__(self, tok, model, adapter_dir=SPACE_LENS):
        super().__init__(tok, model, adapter_dir)
        self.adapter = Path(adapter_dir).name
        self.bullet_prefix = bullet_prefix(adapter_dir)

    def prompt(self, layer):
        return (
            f"An activation vector from layer {layer} of a language model is enclosed in "
            f"activation tags: <activation>㈜</activation>. Produce distinct concepts "
            f"that encode this activation, each as a '- ' bullet on its own line."
        )

    @torch.no_grad()
    def read_layers(self, vecs, sampling=GREEDY, n=1, seed=0):
        """{layer: [d] vector} -> {layer: [n] strings}, all layers decoded in one batch (the
        carrier prompts differ only in the layer number, so they have the same length)."""
        layers = list(vecs)
        carriers = [self._carrier(L) for L in layers]
        if len({(len(i), s) for i, s in carriers}) != 1:
            return {L: self.read(vecs[L], L, sampling, n, seed)[0] for L in layers}
        ids = [c[0] for c in carriers for _ in range(n)]
        v = torch.stack([torch.as_tensor(vecs[L]).float() for L in layers]).to(self.model.device)
        torch.manual_seed(seed)
        with self.lora:
            out = self._generate(ids, carriers[0][1], v.repeat_interleave(n, 0), sampling)
        texts = [self._finish(t, sampling) for t in self.tok.batch_decode(out, skip_special_tokens=True)]
        return {L: texts[i * n : (i + 1) * n] for i, L in enumerate(layers)}

    def _generate(self, ids, slot, vecs, sampling):
        """ids: one carrier (list of ints) shared by every row, or one carrier per row."""
        B = len(vecs)
        rows = ids if isinstance(ids[0], list) else [ids] * B
        ids_t = torch.tensor(rows, device=self.model.device)
        ids = rows[0]
        e = self.model.get_input_embeddings()(ids_t).clone()
        e[:, slot] = (self.alpha * vecs / vecs.norm(dim=-1, keepdim=True).clamp_min(1e-9)).to(e.dtype)
        kw = self._gen_kwargs(sampling, B)
        kw["attention_mask"] = torch.ones(B, len(ids), dtype=torch.long, device=self.model.device)
        return self.model.generate(inputs_embeds=e, **kw)


class CeselderLens(Lens):
    """ceselder/olens-rl-grpo-trajectory-qwen36-27b step_XXXXX. Layer 42 only. Prompt from the
    checkpoint's nla_meta.yaml (`actor`). Injection INFERRED from EasyNLA (karvonen hook):
    at the output of decoder block 1, at the marker (left neighbour 29, right 510),
    h <- h + ||h|| * v/||v||.

    `mode` exists to test that inference: "karvonen" (default), or "embed" (overwrite the input
    embedding with `embed_norm` * v/||v||, the agu18dec style)."""

    layers = (42,)

    def __init__(self, tok, model, step=390, mode="karvonen", embed_norm=None, inject_layer=1, adapter_dir=None, meta_path=None):
        adapter_dir = Path(adapter_dir or CKPT / f"ceselder_olens_rl/step_{step:05d}")
        meta = yaml.safe_load(Path(meta_path or adapter_dir / "nla_meta.yaml").read_text())
        self.template = meta["prompt_templates"]["actor"]
        t = meta["tokens"]
        self.char, self.marker_id = t["injection_char"], int(t["injection_token_id"])
        self.left_id, self.right_id = int(t["injection_left_neighbor_id"]), int(t["injection_right_neighbor_id"])
        assert meta["extraction"]["layer_index"] == 42
        self.mode, self.embed_norm, self.inject_layer = mode, embed_norm, inject_layer
        super().__init__(tok, model, adapter_dir)
        self._vec = None
        self._slot = None
        blocks(model)[inject_layer].register_forward_hook(self._inject)

    def prompt(self, layer):
        return self.template.format(injection_char=self.char)

    def _carrier(self, layer):
        ids, slot = super()._carrier(layer)
        assert ids[slot - 1] == self.left_id and ids[slot + 1] == self.right_id, "marker neighbours drifted"
        return ids, slot

    def _inject(self, mod, inp, out):
        if self._vec is None or self.mode != "karvonen":
            return out
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= self._slot:  # decode step (KV cache): nothing to inject
            return out
        hp = h[:, self._slot].float()
        v = self._vec / self._vec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        h = h.clone()
        h[:, self._slot] = (hp + hp.norm(dim=-1, keepdim=True) * v).to(h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h

    def _generate(self, ids, slot, vecs, sampling):
        B = len(vecs)
        ids_t = torch.tensor([ids] * B, device=self.model.device)
        kw = self._gen_kwargs(sampling, B, skip=0 if self.mode == "embed" else len(ids))
        kw["attention_mask"] = torch.ones_like(ids_t)
        if self.mode == "embed":
            e = self.model.get_input_embeddings()(ids_t).clone()
            norm = self.embed_norm or e[0].norm(dim=-1).mean()
            e[:, slot] = (norm * vecs / vecs.norm(dim=-1, keepdim=True)).to(e.dtype)
            return self.model.generate(inputs_embeds=e, **kw)  # new tokens only
        self._vec, self._slot = vecs, slot
        try:
            return self.model.generate(input_ids=ids_t, **kw)[:, len(ids) :]
        finally:
            self._vec = None
