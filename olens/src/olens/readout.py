"""Full oracle-lens readout for the agu18dec pair: bullets from the AO, each mapped back to a
direction by the AR, then a non-negative least-squares refit in the whitened space.

Conventions verified against the published held-out joint FVE (0.486 here vs 0.487 reported,
scripts/eval_ao.py): the AR head predicts a RAW activation; both target and predictions are
centred with mu and whitened with Sigma^-1/2; a bullet is fed to the AR as the text after
"- " with no leading space.
"""

import re

import torch

from .fve import Whitener, fve_nnls


def bullets(text):
    return [m.strip() for m in re.findall(r"^\s*[-*•]\s+(.+)$", text, re.MULTILINE)]


class Readout:
    def __init__(self, lens, ar):
        self.lens, self.ar = lens, ar
        self._w = {}

    def whitener(self, layer):
        if layer not in self._w:
            self._w[layer] = Whitener(layer, device=self.lens.model.device, dtype=torch.float32)
        return self._w[layer]

    @torch.no_grad()
    def score(self, text, h, layer):
        """One readout text for activation h -> {text, bullets, coefs, fve}, bullets sorted by weight."""
        bl = bullets(text)
        if not bl:
            return {"text": text, "bullets": [], "coefs": [], "fve": 0.0}
        W = self.whitener(layer)
        spans = [self.lens.tok(b, add_special_tokens=False).input_ids[:32] for b in bl]
        fve, c = fve_nnls(W(torch.as_tensor(h)), W(self.ar(spans, layer)))
        order = sorted(range(len(bl)), key=lambda i: -c[i])
        return {"text": text, "bullets": [bl[i] for i in order], "coefs": [float(c[i]) for i in order], "fve": float(fve)}

    def __call__(self, h, layer, sampling=None, n=1, seed=0):
        """One activation -> list of n scored readouts."""
        kw = {} if sampling is None else {"sampling": sampling}
        return [self.score(t, h, layer) for t in self.lens.read(h, layer, n=n, seed=seed, **kw)[0]]

    def layers(self, vecs, sampling=None, n=1, seed=0):
        """{layer: h} -> {layer: [n scored readouts]}, decoded in one batch; every bullet of every
        readout goes through the AR in one forward."""
        kw = {} if sampling is None else {"sampling": sampling}
        texts = self.lens.read_layers(vecs, n=n, seed=seed, **kw)
        items = [(L, t, bullets(t)) for L, ts in texts.items() for t in ts]
        spans = [self.lens.tok(b, add_special_tokens=False).input_ids[:32] for _, _, bl in items for b in bl]
        lays = [L for L, _, bl in items for _ in bl]
        D = self.ar(spans, lays) if spans else None
        out, i = {L: [] for L in texts}, 0
        for L, t, bl in items:
            if not bl:
                out[L].append({"text": t, "bullets": [], "coefs": [], "fve": 0.0})
                continue
            W = self.whitener(L)
            fve, c = fve_nnls(W(torch.as_tensor(vecs[L])), W(D[i : i + len(bl)]))
            i += len(bl)
            order = sorted(range(len(bl)), key=lambda j: -c[j])
            out[L].append({"text": t, "bullets": [bl[j] for j in order], "coefs": [float(c[j]) for j in order], "fve": float(fve)})
        return out
