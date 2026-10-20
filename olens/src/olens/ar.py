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
        layers = layer if isinstance(layer, (list, tuple)) else [layer] * len(spans)
        seqs = [self.tag_ids[self.layers.index(L)].tolist() + list(s) for s, L in zip(spans, layers)]
        n = max(map(len, seqs))
        ids = torch.zeros(len(seqs), n, dtype=torch.long, device=self.model.device)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        with self.lora:
            h = self.model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        h = h[torch.arange(len(seqs)), mask.sum(1) - 1].float()
        return F.layer_norm(h, h.shape[-1:]) @ self.W.T + self.b
