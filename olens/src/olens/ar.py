"""The agu18dec activation reconstructor (AR): `[Layer NN] <span>` -> the predicted RAW residual
activation at the token before the span. Compare in the whitened space of the pooled whitener
(olens.fve.Whitener); see docs/findings.md.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from .lens import CKPT
from .lora import HookLoRA


class AR:
    def __init__(self, model, root=CKPT / "agu18dec_olens_and_ar/ar_ptag_pooled"):
        root = Path(root)
        self.model = model
        self.lora = HookLoRA(model, root / "lora", model.device)
        h = torch.load(root / "heads.pt", map_location=model.device)
        self.W = h["head"]["linear.weight"].float()
        self.b = h["head"]["linear.bias"].float()
        self.tag_ids = h["tag_ids"]
        self.layers = [int(L) for L in h["layers"]]

    @torch.no_grad()
    def __call__(self, spans, layer):
        """spans: list of token-id lists; layer: one layer or one per span (batched in a single
        forward either way). Returns [B, d] head outputs."""
        h, mask = self._forward(spans, layer)
        return self._head(h[torch.arange(len(spans)), mask.sum(1) - 1])

    @torch.no_grad()
    def prefixes(self, spans, layer):
        """Every prefix of each span from the same forward: the AR is causal, so its read at a span's
        k-th token is AR(span[:k]). Returns [B, n, d], n = longest span; rows past a span's end are 0."""
        h, mask = self._forward(spans, layer)
        t = self.tag_ids.shape[1]
        return self._head(h[:, t:]) * mask[:, t:, None]

    def _forward(self, spans, layer):
        layers = layer if isinstance(layer, (list, tuple)) else [layer] * len(spans)
        seqs = [self.tag_ids[self.layers.index(L)].tolist() + list(s) for s, L in zip(spans, layers)]
        n = max(map(len, seqs))
        ids = torch.zeros(len(seqs), n, dtype=torch.long, device=self.model.device)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        with self.lora:
            # no cache: with one, a 256-span batch also builds every layer's linear-attention state (~38 GB)
            h = self.model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        return h, mask

    def _head(self, h):
        h = h.float()
        return F.layer_norm(h, h.shape[-1:]) @ self.W.T + self.b
