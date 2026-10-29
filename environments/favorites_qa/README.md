# favorites_qa: Qwen's own preferences, free response

**Status (2026-10-29):** ACTIVE. Bank v3 (5,187 questions, 1,500 stable under Qwen); v2 and the
pilot v1 are kept. One arm, no hints. Everything else is in
[docs/envs/favorites_qa.md](../../docs/envs/favorites_qa.md).

|       |                                                                                                                                        |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Code  | [`env.py`](env.py); bank on HF (`banks/favorites_qa/v3.json`; `FAVORITES_QA_BANK=v2` / `v1` for the older banks), fetched on first use |
| Arms  | `nohint`: the question as the only user message, no system prompt, thinking off, T 1.0 / top_p 1.0                                     |
| Items | the stable (≥ 7/8) items; `FAVORITES_QA_ALL=1` for every item                                                                          |
| Label | `qwen_modal` (a raw sample of Qwen's gated modal answer); `measure()` gives `modal`, `contains_modal`                                  |

```bash
uv run python -m environments run favorites_qa --tag dev --items 0:10 --n 4
```

## Runs

`favorites_qa/2026-10-29_pilot62-nohint`: the 62 stable v1 items, n = 1, the prompts of the
favorites_qa_inline pilot's end-of-prompt O-lens reads. `favorites_qa/2026-10-29_v3-1500-nohint`:
all 1,500 stable v3 items, n = 1, the prompts of the v3 O-lens / NLA / J++ reads.
