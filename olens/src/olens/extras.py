"""Two more readouts on the shared base: the ceselder L42 natural-language autoencoder (NLA) and the
Ayonrinde & Lindsey J++ lens.

NLA (ceselder/qwen3.6-27b-nla-L42, EasyNLA): the AV is a LoRA that reads one activation injected
Karvonen-style (h + ||h|| v/||v|| at the output of block 1, at the ㈜ marker of the `actor` prompt)
and writes an <explanation>. The AR (`rl_critic_step400`) is the base truncated to 43 blocks with
no final norm; its last-token hidden state, scaled to norm sqrt(d), goes through a fp32
Linear(d, d) value head. The critic prompt is the raw `critic` template (no chat template). FVE is
the paper's: 1 - ||p - v||^2 / E||v - mu||^2 with p, v scaled to norm sqrt(d) and mu, E over a
calibration set (`calibrate`).

J++ (neuronpedia/jacobian-lens qwen3.6-27b/jpp): logits at layer l = lm_head(norm(h_l @ J[l].T)),
h_l the output of block l (= hidden_states[l + 1]), l in 0..62.
"""

import math
import re
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from transformers import StoppingCriteria, StoppingCriteriaList

from .lens import CeselderLens, Sampling, capture
from .lora import HookLoRA

NLA_REPO = "ceselder/qwen3.6-27b-nla-L42"
JPP_REPO, JPP_FILE = "neuronpedia/jacobian-lens", "qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt"
EXPL = re.compile(r"<explanation>(.*?)(?:</explanation>|$)", re.S)


class _StopOn(StoppingCriteria):
    def __init__(self, tok, s):
        self.tok, self.s = tok, s

    def __call__(self, input_ids, scores, **kw):
        tail = self.tok.batch_decode(input_ids[:, -6:])
        return torch.tensor([self.s in t for t in tail], device=input_ids.device)


class NLA(CeselderLens):
    """The AV. `stack` = also apply av_sft_lora under the RL LoRA (its adapter_config names an
    SFT-merged base); False = the RL LoRA alone on the base."""

    def __init__(self, tok, model, stack=True, local_dir=None):
        d = Path(local_dir or snapshot_download(NLA_REPO, allow_patterns=["nla_meta.yaml", "av_*/*"]))
        super().__init__(tok, model, adapter_dir=d / "av_rl_lora_step400", meta_path=d / "nla_meta.yaml")
        self.sft = HookLoRA(model, d / "av_sft_lora", model.device)
        self.stack = stack

    @torch.no_grad()
    def explain(self, vecs, max_new_tokens=220, temperature=0.0, batch=64):
        """[B, d] raw L42 activations -> [B] explanation strings."""
        out = []
        ids, slot = self._carrier(42)
        smp = Sampling(temperature=temperature, max_new_tokens=max_new_tokens)
        for i in range(0, len(vecs), batch):
            v = vecs[i : i + batch].to(self.model.device).float()
            B = len(v)
            ids_t = torch.tensor([ids] * B, device=self.model.device)
            kw = dict(
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                attention_mask=torch.ones_like(ids_t),
                stopping_criteria=StoppingCriteriaList([_StopOn(self.tok, "</explanation>")]),
                **smp.kwargs(),
            )
            self._vec, self._slot = v, slot
            self.sft.active = self.stack
            try:
                with self.lora:
                    g = self.model.generate(input_ids=ids_t, **kw)[:, len(ids) :]
            finally:
                self._vec, self.sft.active = None, False
            for t in self.tok.batch_decode(g, skip_special_tokens=True):
                m = EXPL.search(t)
                out.append((m.group(1) if m else t).strip())
        return out


class NLACritic(nn.Module):
    """The AR: explanation text -> predicted L42 activation direction (norm sqrt(d))."""

    def __init__(self, tok, device="cuda", sub="rl_critic_step400", template=None):
        super().__init__()
        from transformers import AutoModelForCausalLM

        d = Path(snapshot_download(NLA_REPO, allow_patterns=[f"{sub}/*", "nla_meta.yaml"]))
        import yaml

        meta = yaml.safe_load((d / "nla_meta.yaml").read_text())
        self.template = template or "Summary of the following text: <text>{explanation}</text> <summary>"
        assert meta["prompt_templates"].get("critic", self.template) == self.template
        m = AutoModelForCausalLM.from_pretrained(d / sub, dtype=torch.bfloat16).to(device)
        m.eval().requires_grad_(False)
        assert len(m.model.layers) == 43
        m.model.norm = nn.Identity()
        m.lm_head = nn.Identity()
        self.body = m.model
        self.head = nn.Linear(5120, 5120, bias=False).to(device, torch.float32)
        self.head.load_state_dict(load_file(d / sub / "value_head.safetensors"))
        self.tok, self.scale = tok, math.sqrt(5120)
        self.mu, self.base = None, None

    @torch.no_grad()
    def predict(self, texts, batch=32):
        out = []
        for i in range(0, len(texts), batch):
            rows = [self.tok(self.template.format(explanation=t), add_special_tokens=False).input_ids for t in texts[i : i + batch]]
            T = max(map(len, rows))
            pad = self.tok.pad_token_id or 0
            ids = torch.tensor([[pad] * (T - len(r)) + r for r in rows], device=self.head.weight.device)  # left pad
            am = torch.tensor([[0] * (T - len(r)) + [1] * len(r) for r in rows], device=ids.device)
            h = self.body(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state[:, -1].float()
            out.append(self.head(norm_to(h, self.scale)))
        return norm_to(torch.cat(out), self.scale)

    def calibrate(self, vecs):
        """mu and the predict-the-mean MSE over raw L42 activations [N, d] (the FVE denominator)."""
        v = norm_to(vecs.float().to(self.head.weight.device), self.scale)
        self.mu = v.mean(0)
        self.base = float(((v - self.mu) ** 2).sum(-1).mean())

    @torch.no_grad()
    def score(self, texts, vecs):
        """-> [{cos, fve}] per (text, raw activation)."""
        p = self.predict(texts)
        v = norm_to(vecs.float().to(p.device), self.scale)
        cos = (p * v).sum(-1) / self.scale**2
        mse = ((p - v) ** 2).sum(-1)
        fve = 1 - mse / self.base if self.base else torch.full_like(mse, float("nan"))
        return [{"cos": round(float(c), 4), "fve": round(float(f), 4)} for c, f in zip(cos, fve)]


def norm_to(v, s):
    return v * (s / v.norm(dim=-1, keepdim=True).clamp_min(1e-12))


class JPP:
    """J++ lens over every source layer 0..62."""

    def __init__(self, model, device="cuda"):
        d = torch.load(hf_hub_download(JPP_REPO, JPP_FILE), map_location="cpu", weights_only=True)
        self.layers = sorted(int(L) for L in d["J"])
        self.J = torch.stack([d["J"][L] for L in self.layers]).to(device)  # [63, d, d] fp16
        self.model = model

    @torch.no_grad()
    def read(self, ids, positions, k=10, chunk=16):
        """-> {pos: [[(token_id, prob)] * k per layer]} at every source layer."""
        acts = capture(self.model, ids, self.layers)
        H = torch.stack([acts[L][positions] for L in self.layers])  # [63, P, d] fp32
        del acts
        res = {p: [] for p in positions}
        for a in range(0, len(self.layers), chunk):
            x = torch.einsum("lpd,led->lpe", H[a : a + chunk], self.J[a : a + chunk].float())
            lg = self.model.lm_head(self.model.model.norm(x.to(torch.bfloat16))).float()
            pr = lg.softmax(-1)
            top = pr.topk(k, dim=-1)
            for li in range(x.shape[0]):
                for j, p in enumerate(positions):
                    res[p].append([[int(t), round(float(q), 4)] for t, q in zip(top.indices[li, j], top.values[li, j])])
        return res
