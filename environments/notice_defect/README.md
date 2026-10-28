# notice_defect (latent noticing of a planted defect)

**Status (2026-10-27):** ACTIVE (subject side). v2 after the audit: 58 items, clean twins,
artifact-first prompts, text-only gate. Design, per-item results and the gate log are in
[docs/envs/notice_defect.md](../../docs/envs/notice_defect.md).

Each item is a short non-code artifact with ONE planted defect (one `{x}` span: `bad`, or `good` for
the clean twin). The artifact comes first in the user turn, the instruction after it, so the
artifact tokens are identical across arms. Arms (factors `instruction` × `artifact`):

- `task`: a task orthogonal to the defective field (subject line / audience / title and tags / tone
  and purpose) on the defective artifact. **The rollout the O-lens is read on.**
- `ask`: "Are there any errors, mistakes, or problems with it? … if not, say it looks correct." The
  modal `ask` label is the **noticing oracle** per item.
- `task_clean`, `ask_clean`: the same on the clean twin (lens control; `ask_clean` = how often Qwen
  claims an error when nothing is planted, which it does on some correct artifacts).
- `ask_v2`: `ask` without the scripted exit (flags far more; not the oracle).

|                 |                                                                                                                                            |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Code            | [`env.py`](env.py), [`gate.py`](gate.py)                                                                                                   |
| Items           | 58: `dow` 10, `arith` 12, `calendar` 5, `contra` 11, `fact` 11, `physical` 9 (`ITEMS`, `TYPES`)                                            |
| Primary measure | `positive` = `flagged`: a sentence names the defective value or its correction with an error word and no negation, or an item `flag` regex |
| Pairs           | `notice`: `task` → `ask`; `notice_clean`: `task_clean` → `ask_clean`                                                                       |
| Requires        | thinking off                                                                                                                               |

```bash
uv run python -m environments run notice_defect --tag smoke --n 8                 # 58 x 5 arms x 8, ~$1.20
uv run python -m environments.notice_defect.gate <run>[+<top-up>] --slug flash3     # Flash text-only gate, 174 requests
```

Defaults match hint_mcq: Qwen3.6-27B on OpenRouter, `deepinfra/fp8`, thinking off, T 0.7 / top-p 0.8
/ presence 1.5 (try 1 ran on `venice/fp8`, see the doc).

## Runs

- `2026-10-27_try1`: 58 items × 5 arms, n ≈ 5 (402 mid-run); 40 usable items, 9 not noticed + 31
  noticed; task arm 0/280 verbalized. Gate pending (Gemini quota).
