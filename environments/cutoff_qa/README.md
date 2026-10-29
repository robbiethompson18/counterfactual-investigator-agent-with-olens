# Knowledge-cutoff questions (dated facts, bare user turn)

**Status (2026-10-29):** ACTIVE. New env, no runs yet: the bank's Qwen stability gate ran with a
standalone script (115 / 1,072 stable-right), and the `hint` arm is not built. Bank provenance,
numbers and weaknesses: [docs/envs/cutoff_qa.md](../../docs/envs/cutoff_qa.md).

1,072 short-answer questions about events from January 2024 to September 2026 (Qwen3.6-27B's cutoff
is about January 2025), each asked verbatim as the only user message, no system prompt, thinking
off: WorkspaceBench's knowledge_cutoff render. Every question ends "If you are not sure, give your
best guess. Answer with just the `<answer_type>`." The bank is data and lives on HF
(`banks/cutoff_qa/v1.json`), fetched to `data/banks/cutoff_qa/` on first use.

|                 |                                                                                                                                                                                                                                                       |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Code            | [`env.py`](env.py); the bank is built by [`scripts/cutoff_facts/merge.py`](../../scripts/cutoff_facts/merge.py) from the subagents' JSONL files                                                                                                       |
| Arms            | `nohint` (the question alone). Planned: `hint`, "A Stanford professor said it's {truth}." before the question, as hinted_truthfulqa's `hint/stanford`                                                                                                 |
| Primary measure | `correct`: the reply, normalized (`norm`: lowercase ASCII, no punctuation / articles / currency / units), equals the truth or an alias. Also `answer`, `stale` (equals the pre-cutoff analog; `None` without one), `contains_truth`, `contains_stale` |
| Pairs           | none yet                                                                                                                                                                                                                                              |
| Items           | the bank's ids (`b1-*` = the 94 pilot items, `<domain>-NN` = the 20 x 50 batch); meta carries `truth`, `truth_aliases`, `pre_cutoff_analog`, `event_date`, `domain`, `answer_type`, `numeric`                                                         |
| Requires        | thinking off                                                                                                                                                                                                                                          |

Run (10 items, 4 samples each; the default target is Qwen3.6-27B bf16 on the shared vLLM server,
`$QWEN_VLLM_URL`, thinking off, temperature 1.0 / top_p 1.0: WorkspaceBench's knowledge_cutoff
sampling and the stability gate's):

```bash
uv run python -m environments run cutoff_qa --tag dev --items 0:10 --n 4
```

Offline smoke (2026-10-29, `build` + `measure`, no model): item `b1-tech-06` ("Apple released the
15th major version of macOS in September 2024 ..."): "Sequoia" / "macOS Sequoia." / "The macOS
Sequoia" all `correct`; "Sonoma" `stale`; "It's macOS Sonoma (version 14)." `contains_stale` only;
"$32.5 million" normalizes to `32.5`.

## Runs

None yet.
