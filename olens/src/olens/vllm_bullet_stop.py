"""vLLM (v1) logits processor that ends a lens read once it holds k complete bullets, like olens.lens.BulletStop does for
HF generate. Requests opt in with `SamplingParams(extra_args={"max_bullets": k})`; rows without it are untouched.

Pass the class to `LLM(logits_processors=[BulletStop])`; vLLM builds one per engine process (so this module must be
importable there: PYTHONPATH=olens/src). A bullet starts at the first token of a line whose text, after whitespace,
begins with `-`, `*` or `•` (olens.lens.BULLET); the (k+1)-th such start forces <|im_end|> on the next step, and
`trim_bullets` drops that partial line.
"""

import torch
from transformers import AutoTokenizer
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor, MoveDirectionality

BULLET_CHARS = ("-", "*", "•")


class BulletStop(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        tok = AutoTokenizer.from_pretrained(vllm_config.model_config.tokenizer)
        self.eos = tok.eos_token_id
        self._text = {}  # token id -> decoded text, memoised per token
        self._tok = tok
        self.rows: dict[int, dict] = {}  # batch index -> {k, out (live list), n bullets, scanned, line_start}

    def is_argmax_invariant(self) -> bool:
        return False

    def _t(self, i):
        if i not in self._text:
            self._text[i] = self._tok.decode([i])
        return self._text[i]

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.rows.pop(index, None)
        for added in batch_update.added:
            index, params, out = added[0], added[1], added[-1]
            k = (getattr(params, "extra_args", None) or {}).get("max_bullets")
            if k:
                self.rows[index] = {"k": int(k), "out": out, "n": 0, "scanned": 0, "ls": True}
            else:
                self.rows.pop(index, None)
        for a, b, how in batch_update.moved:
            ra, rb = self.rows.pop(a, None), self.rows.pop(b, None)
            if how == MoveDirectionality.SWAP and rb is not None:
                self.rows[a] = rb
            if ra is not None:
                self.rows[b] = ra

    def _scan(self, st):
        out = st["out"]
        for i in range(st["scanned"], len(out)):
            text = self._t(out[i])
            # a bullet starts at a line start: the first non-blank char after a newline (or at position 0) is a marker
            for piece_idx, piece in enumerate(text.split("\n")):
                at_line_start = st["ls"] if piece_idx == 0 else True
                if at_line_start and piece.strip() and piece.lstrip()[0] in BULLET_CHARS:
                    st["n"] += 1
                if piece.strip():
                    st["ls"] = False
                if piece_idx < text.count("\n"):
                    st["ls"] = True
            st["scanned"] = i + 1

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for index, st in self.rows.items():
            self._scan(st)
            if st["n"] > st["k"] and index < logits.shape[0]:
                logits[index].fill_(float("-inf"))
                logits[index, self.eos] = 0.0
        return logits
