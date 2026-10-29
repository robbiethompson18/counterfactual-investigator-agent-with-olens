# cutoff_qa: present-tense factual questions across Qwen's knowledge cutoff

**Status (2026-10-29):** ACTIVE. Bank v4 (6,411 questions, 2,605 stable under Qwen, 861 of them
stale) is the default. One arm, no hints. Everything else is in
[docs/envs/cutoff_qa.md](../../docs/envs/cutoff_qa.md); results in
[docs/cutoff-qa-inline.md](../../docs/cutoff-qa-inline.md).

|       |                                                                                                                    |
| ----- | ------------------------------------------------------------------------------------------------------------------ |
| Code  | [`env.py`](env.py); banks on HF (`banks/cutoff_qa/v4.json`), fetched on first use                                  |
| Arms  | `nohint`: the question as the only user message, no system prompt, thinking off, T 1.0 / top_p 1.0                 |
| Items | the stable (≥ 7/8) items; `CUTOFF_QA_ALL=1` for every item, `CUTOFF_QA_BANK=v3` / `v2` / `v1` for older banks      |
| Label | `qwen_modal` (Qwen's gated modal answer); `measure()` gives `correct`, `stale`, `contains_truth`, `contains_stale` |

```bash
uv run python -m environments run cutoff_qa --tag dev --items 0:10 --n 4
```

## Runs

`cutoff_qa/2026-10-29_smoke200-nohint`: the first 200 stable non-numeric v2 items, n = 1, for the
investigator smoke in docs/cutoff-qa-inline.md.
