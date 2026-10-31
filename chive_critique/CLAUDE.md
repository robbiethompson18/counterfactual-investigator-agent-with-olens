# chive_critique

Numbers behind [docs/chive-critique.md](../docs/chive-critique.md), our critique of the CHIVE dataset
(Karvonen et al., 2026). Read the doc first; this is how to reproduce it.

```bash
uv run python -m chive_critique          # everything, about a minute after the first download
uv run python -m chive_critique adam     # Adam's release only
uv run python -m chive_critique agam     # Agam's 300-incident run only
```

Data, all public:

- **Adam's release** [adamkarvonen/chive-data](https://huggingface.co/datasets/adamkarvonen/chive-data),
  pinned to the revision in `data.py`: prompts, claim banks, eval sets and the scored evaluator arms
  of the three main-body WildChat eval runs (Gemma 3 27B, Qwen3-8B, Qwen3.5-397B). Downloaded on
  first use into `data/.cache/chive_adam/` (~700 MB).
- **Our run dir** `runs/chive_critique/2026-10-30_replication-data/` in the sprint dataset
  [Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens),
  pulled on first use: `adam_rates/` (per-experiment k/n streamed out of Adam's 1.5 GB
  `investigations.jsonl` files, so nobody streams them again; `python -m chive_critique extract`
  regenerates them), `adam_findings/` (each investigation's structured findings, for the
  contradiction grading), `agam_inert_labels.jsonl` and `adam_inert_labels.jsonl` (hand labels of
  inert ruled-out items: Sonnet subagents graded, the main session adjudicated).
- **Agam's run** [agu18dec/chive-forensics](https://huggingface.co/datasets/agu18dec/chive-forensics)
  and his frozen held-out scores (`viewer/cmp_climb_test.json` in our dataset).

`data.py` holds every path and revision; `adam.py` and `agam.py` print one section each. No LLM is
called: the hand labels are data, re-grading is a Claude Code subagent job, not an API call.
