# knowledge_cutoff

**Status (2026-10-29):** ACTIVE. WorkspaceBench's `knowledge_cutoff` questions as an env, so Qwen's
samples, other models' answers and O-lens reads are ordinary runs; the investigator task is the
cfbench eval `knowledge_cutoff`. Decisions and results:
[docs/envs/knowledge_cutoff.md](../../docs/envs/knowledge_cutoff.md).

|                 |                                                                                                                                                                                      |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Code            | [`env.py`](env.py); scorer is the ported WorkspaceBench alias regex in `wsb/knowledge_cutoff.py`                                                                                     |
| Items           | the 109 frozen items of WorkspaceBench's family (bank on HF `banks/wsb/knowledge_cutoff.json`, dev only); Sonnet-authored 2024–2026 questions gated on Qwen are the planned main set |
| Arms            | `recall` (their template, "Answer with just the <answer_type>."); no MCQ: shown options, Qwen switches to the true one on a third of items                                           |
| Primary measure | `stale` = the reply names the bank's gold, Qwen3.6-27B's own stable answer; also `answer`, `correct` (= the truth)                                                                   |
| Defaults        | `qwen/qwen3.6-27b` on OpenRouter (deepinfra/fp8), thinking off, T 1 top-p 1 (WorkspaceBench's gate sampling)                                                                         |

```bash
uv run python -m environments run knowledge_cutoff --tag dev109-recall --arms recall --n 8 --conc 2   # OpenRouter caps Qwen at 20/min
uv run python -m environments run knowledge_cutoff --tag census-haiku --arms recall --n 1 --model anthropic/claude-haiku-5.5 --backend openrouter --provider any --thinking on
```

A run of another model as the subject is the investigator census: its `stale` rate is how often its
own answer is Qwen's stale one.

## Runs

| run                        | target                                                     | episodes | note                            |
| -------------------------- | ---------------------------------------------------------- | -------- | ------------------------------- |
| `2026-10-29_dev109-recall` | `qwen/qwen3.6-27b@openrouter:deepinfra/fp8` (thinking off) | 872      | the 109 dev items × 8, `recall` |
