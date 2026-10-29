# Knowledge-cutoff questions (present tense, bare user turn)

**Status (2026-10-29):** ACTIVE. New env, no runs yet. Default bank v2: 803 present-tense questions
("Who is the CEO of Intel?") gated on Qwen, items = the 266 stable ones (187 stable-wrong: Qwen's
stale answer). One arm, no hints. Provenance, the gate numbers and the present-tense finding:
[docs/envs/cutoff_qa.md](../../docs/envs/cutoff_qa.md).

Each question is asked verbatim as the only user message, no system prompt, thinking off:
WorkspaceBench's knowledge_cutoff render and sampling (T 1.0, top_p 1.0). The label for the
predict-Qwen task is the gate's `qwen_modal`, not the truth. Banks are data and live on HF
(`banks/cutoff_qa/v1.json`, `v2.json`), fetched to `data/banks/cutoff_qa/` on first use;
`CUTOFF_QA_BANK=v1` loads the dated v1 bank, `CUTOFF_QA_ALL=1` includes unstable and refusal items.

|                 |                                                                                                                                                                                                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Code            | [`env.py`](env.py); the bank is built by [`scripts/cutoff_facts/merge.py`](../../scripts/cutoff_facts/merge.py) from the subagents' JSONL files                                                                                                                           |
| Arms            | `nohint` (the question alone); hints are ruled out for this env                                                                                                                                                                                                           |
| Primary measure | `correct`: the reply, normalized (`norm`: lowercase ASCII, no punctuation / articles / currency / units), equals the truth or an alias. Also `answer`, `stale` (equals Qwen's gated `qwen_modal`; for v1 items the pre-cutoff analog), `contains_truth`, `contains_stale` |
| Pairs           | none (one arm)                                                                                                                                                                                                                                                            |
| Items           | v2's stable items by default (ids as in v1: `b1-*` pilot, `<domain>-NN` batch, `f-*` fame-steered); meta carries the truth fields, `qwen_modal`, `gate_class`, `truth_now`, `numeric` and `bank`                                                                          |
| Requires        | thinking off                                                                                                                                                                                                                                                              |

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
