# Code smells

Tech debt we know about, newest first.

## 2026-10-28: metamodel reader bugs from a Codex review of the serving code

A Codex read of `olens/src/olens/`, `scripts/olens_read_requests.py`,
`scripts/jpp_read_requests.py`, `src/.../render.py` and `runpod/olens_reads.sh` (no GPU run). It
found no layer-index, norm or position-alignment bug in the J++ or O-lens paths. It also found
these (its seed, thinking-render, resume and stale-LoRA bugs are fixed and removed):

- **Field matching can run into a later message** (`render.py` ~78-88): `text.find(s, cur)` searches
  the whole rest of the transcript, so a short field (an assistant's `"A"`) can match inside a later
  message and shift every field after it; an answer `"think"` matches inside `<think>`. Single-turn
  envs are safe; multi-turn ones aren't. Fix: get field spans from the renderer, or match within
  message boundaries.
- Smaller: template tokens carry the previous message's id/role (cfbench places them itself since
  harness v8); `char_offset` ignores a field's stripped leading whitespace (reads land a few chars
  early); per-token decoding mangles characters split across tokens in saved transcript text; J++
  provenance records a blob hash, not the commit, and J++ ignores per-request `layers`.
