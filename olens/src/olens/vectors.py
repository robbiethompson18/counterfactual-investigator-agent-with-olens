"""Probe / steering directions in the residual stream after block L, built from the lenses, and a
steering hook.

Specs (JSON-able dicts):
  {"kind": "jpp", "layer": L, "plus": [str], "minus": [str]}
      the direction that raises the J++ logits of `plus` (minus those of `minus`) at layer L:
      J[L]^T (g * W_U[t]) summed over tokens, g the final norm's gain. Each string's first token.
  {"kind": "ar", "layer": L, "text": str}     agu18dec AR prediction for `text`, minus the whitener mean
  {"kind": "nla", "text": str}                NLA reconstructor (L42) prediction, minus the calibration mean
  {"kind": "diff", "a": spec, "b": spec}      a - b (same layer)
Every result is a unit vector [d] plus its layer.
"""

import torch

from .lens import blocks


class Vectors:
    def __init__(self, tok, model, readout=None, jpp=None, critic=None):
        self.tok, self.model, self.readout, self.jpp, self.critic = tok, model, readout, jpp, critic

    def first_token(self, s):
        ids = self.tok(s, add_special_tokens=False).input_ids
        assert ids, f"empty token string {s!r}"
        return ids[0]

    @torch.no_grad()
    def build(self, spec):
        """-> (unit vector [d] fp32 on the model device, layer, info)"""
        k = spec["kind"]
        if k == "diff":
            a, La, ia = self.build_raw(spec["a"])
            b, Lb, ib = self.build_raw(spec["b"])
            assert La == Lb, f"diff of vectors at different layers ({La}, {Lb})"
            return unit(a - b), La, {"a": ia, "b": ib, "cos_ab": round(float(unit(a) @ unit(b)), 4)}
        v, L, info = self.build_raw(spec)
        return unit(v), L, info

    @torch.no_grad()
    def build_raw(self, spec):
        k = spec["kind"]
        if k == "jpp":
            L = int(spec["layer"])
            i = self.jpp.layers.index(L)
            g = self.model.model.norm(torch.ones(1, self.model.config.hidden_size, device=self.model.device, dtype=torch.bfloat16)).float()[0]
            W = self.model.lm_head.weight
            plus = [self.first_token(s) for s in spec.get("plus") or []]
            minus = [self.first_token(s) for s in spec.get("minus") or []]
            u = torch.zeros_like(g)
            if plus:
                u += W[plus].float().mean(0)
            if minus:
                u -= W[minus].float().mean(0)
            v = self.jpp.J[i].float().T @ (g * u)
            return v, L, {"plus": [self.tok.decode([t]) for t in plus], "minus": [self.tok.decode([t]) for t in minus]}
        if k == "ar":
            L = int(spec["layer"])
            ids = self.tok(spec["text"], add_special_tokens=False).input_ids[:32]
            p = self.readout.ar([ids], L)[0]
            return p - self.readout.whitener(L).mu.to(p.device, p.dtype).view(-1), L, {"tokens": len(ids)}
        if k == "nla":
            p = self.critic.predict([spec["text"]])[0]
            return p - self.critic.mu, 42, {}
        raise ValueError(f"unknown vector kind {k!r}")

    @torch.no_grad()
    def probe(self, ids, layer, v):
        """cosine of the block-`layer` residual with v at every position, and the projection."""
        from .lens import capture

        h = capture(self.model, ids, [layer])[layer]
        v = unit(v.to(h.device))
        proj = h @ v
        return {"cos": [round(float(x), 4) for x in proj / h.norm(dim=-1).clamp_min(1e-9)], "proj": [round(float(x), 2) for x in proj]}

    @torch.no_grad()
    def steer(self, ids, layer, v, alpha, n=64, max_new_tokens=400, temperature=1.0, top_p=1.0, seed=0, batch=64):
        """Sample n replies to the prompt `ids` (ending in the generation prompt) with
        h <- h + alpha * ||h|| * v at the output of block `layer`, every position."""
        v = unit(v.to(self.model.device)).to(torch.bfloat16)

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + alpha * h.norm(dim=-1, keepdim=True) * v
            return (h, *out[1:]) if isinstance(out, tuple) else h

        hd = blocks(self.model)[layer].register_forward_hook(hook) if alpha else None
        texts = []
        try:
            torch.manual_seed(seed)
            for i in range(0, n, batch):
                B = min(batch, n - i)
                x = torch.tensor([ids] * B, device=self.model.device)
                out = self.model.generate(
                    input_ids=x,
                    attention_mask=torch.ones_like(x),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=0,
                    pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                )
                texts += self.tok.batch_decode(out[:, len(ids) :], skip_special_tokens=True)
        finally:
            if hd:
                hd.remove()
        return texts


def unit(v):
    return v / v.norm().clamp_min(1e-12)
