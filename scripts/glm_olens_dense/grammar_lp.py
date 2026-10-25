"""Batched version of Celeste's one-bullet grammar v3 logits processor (olens_glm53_usage.GlmBulletGrammarLP).

Same masks and the same per-position category rule (olens_glm53_usage.grammar_masks / grammar_cat), but applied to the whole batch with one
gather + add per step instead of one Python closure call per request, which bounded the per-request version at ~25 positions/s on 4xB200.
A request opts in exactly as before: extra_args={"bullet_grammar": {"cap": 16, "fixed": False}}.
"""

from __future__ import annotations

import olens_glm53_usage as U
import torch
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor, MoveDirectionality


class BatchedBulletGrammarLP(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(vllm_config.model_config.tokenizer)
        self.device = device
        self.G: torch.Tensor | None = None
        self.state: dict[int, tuple[U.GrammarSpec, list[int]]] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return
        for i in batch_update.removed:
            self.state.pop(i, None)
        for i, params, _prompt, out_ids in batch_update.added:
            cfg = (getattr(params, "extra_args", None) or {}).get("bullet_grammar") if params is not None else None
            if cfg:
                self.state[i] = (U.GrammarSpec(cap=int(cfg.get("cap", 16)), fixed=bool(cfg.get("fixed", False))), out_ids)
            else:
                self.state.pop(i, None)
        for a, b, kind in batch_update.moved:
            sa, sb = self.state.pop(a, None), self.state.pop(b, None)
            if sa is not None:
                self.state[b] = sa
            if kind == MoveDirectionality.SWAP and sb is not None:
                self.state[a] = sb

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.state:
            return logits
        G = self.G
        if G is None or G.shape[1] != logits.shape[-1] or G.device != logits.device:
            G = self.G = U.grammar_masks(self.tok, logits.shape[-1], logits.device)
        rows, cats = [], []
        for i, (spec, out) in self.state.items():
            if i < logits.shape[0]:
                rows.append(i)
                cats.append(U.grammar_cat(len(out), spec, out[-1] if out else None))
        r = torch.tensor(rows, device=logits.device)
        c = torch.tensor(cats, device=logits.device)
        logits[r] = logits[r] + G[c].to(logits.dtype)
        return logits
