# wsb: WorkspaceBench families as text-only baselines

Runs a plain model (no lens, no activations) on a WorkspaceBench family and scores it with
WorkspaceBench's own judge, so the result is comparable with the O-lens numbers in
[camilablank/workspace-bench](https://github.com/camilablank/workspace-bench) (MIT, Camila Blank).
`user_modeling`, `jailbreak_recognition`, `basic_readout`, `knowledge_cutoff`, `wrong_bridge` and `moral_rationale` so far.

```bash
source .envrc.local && uv run python -m wsb.user_modeling --n 3 --limit 10   # 5 selfdescribe + 5 synthsys, bank order
uv run python -m wsb.user_modeling --items um-sd-country-00,um-ss-continent-02
uv run python -m wsb.user_modeling --n 3 --subfamily synthsys                # the 53 synthsys items
```

- **Banks are data: they live on HF, never in git.** `banks.load(family)` reads
  `data/banks/wsb/<family>.json`, downloading it from `banks/wsb/` in the sprint HF dataset on first
  use, so any checkout works. `build_banks.py` rebuilds them from the pinned
  `vendor/workspace-bench` submodule (their `evals/<family>/items.json` verbatim under a `source`
  header) and `--upload`s them. Reading workspace-bench's data files is fine; never import or run
  its code.
- `user_modeling.py`: the responder (`claude-sonnet-5-5` on the Anthropic API, effort `medium`,
  adaptive thinking, `--n` samples per item, each item's system + user turn sent unchanged), scoring
  and the CLI. Every run goes through `runs.py` (`data/runs/wsb_user_modeling/<date>_<slug>/`:
  `responses.jsonl`, `judged.jsonl`, `summary.json`, `meta.json`, `expview/`) and is synced to HF at
  the end.
- `judge.py`: their 6-way identification MC, ported faithfully: gold + 4 same-class distractors
  (seeded draw and shuffle from the whole bank, topped up from `UM_OPTION_SUPPLEMENT`) + "cannot
  tell from the readout", prompt `um-v2`, their JSON schema; checked item-for-item against their
  `tests/golden/user_modeling_options.json` and `user_modeling_prompt.txt`. Also a cheap regex hit:
  an `intermediates` alias in the reply (word boundary, case-insensitive).

**The judge is theirs, not ours:** `claude-opus-5-5` at effort `low` with structured output, the
judge and settings of the only public per-item lens rows (`agu18dec/olens_ablations`
`rl.base.off_k0.own.i300`, um-v2, seed 0) and of their own Anthropic route. The repo default judge
(Haiku 5.5, medium) would make the numbers incomparable. The post's own headline used Gemini 3.8
Flash (minimal); that route was dropped on 2026-10-28 when our Gemini billing died. The judge prompt
talks about an "oracle lens" readout; here the "readout" is the model's reply, which is exactly the
text-only floor the lens should beat.

Scoring (`summary.json`, printed as a table by subfamily and overall): per item `pass@1` = mean over
samples, `pass@n` = any sample; `inferred` = picked gold with basis `inferred_characterization`
(their headline), `correct` = picked gold regardless of basis, `regex` = alias hit. Refusals
(responder or judge) score as fails and are counted under `refused`. Token counts are summed from
`usage`.

`jailbreak.py` replays their anti-text-inversion gate for `jailbreak_recognition` rather than having
the model answer the chat: Sonnet 5.5 is shown the Qwen3.6-27B chat render through the last user
turn's `<|im_end|>` and asked "what would a language model be thinking about on this token?" (their
`open_model` framing, API-default thinking as their gate called it), and their `jb-v1` judge (Opus
5.5, the gate's judge) labels the answer as one readout. Its HF bank adds that render per item
(`prefix`, `token`), made by `build_banks.py` with the real tokenizer, which asserts the token
positions against their bank. Prompts are byte-identical to theirs. pass@1 is reported with refusals
dropped (their rule), as pass and as fail. Their parser drops 0-based verdict indices, which we copy
faithfully; see camila-blank/workspace-bench#107.

```bash
uv run python -m wsb.jailbreak --n 3        # all 86 items x 3 draws, ~5 min
```

`basic_readout.py` has Sonnet 5.5 complete all 100 items (35 entity "The athlete Muhammad Ali plays
the sport of", 33 processing "10 - 1 =", 32 implicit "What is your favorite card game? Answer in one
word.", gold = Qwen3.6-27B's gated answer) with `--persona qwen` (system prompt "Answer as if you
are Qwen 3.6 27B") or `--persona self` (no system prompt). Implicit items get their answer stem "My
one-word answer is:" at the end of the user turn, since Sonnet with thinking refuses an assistant
prefill. Their single-token bank judge (`bank-2026-10-10`, pasted verbatim in the module,
quote-verified) runs on Opus 5.5 at effort low; `lens` is the per-item verdict of the lens arm
(`agu18dec/olens_ablations` `rl.base.off_k0.own.i300`, any of 11 layers). `summary.json` has
judge/regex pass@1 and pass@n and the lens rate by subfamily.

```bash
uv run python -m wsb.basic_readout --persona qwen --n 5 --slug sonnet-as-qwen   # 500 Sonnet + 500 Opus calls
uv run python -m wsb.basic_readout --persona self --n 5 --slug sonnet-self
```

`moral_rationale.py` is the same gate replay for `moral_rationale`: the prefix is the whole chat
render of the ethical yes/no stimulus (through the generation prompt, the last of the family's 5
read positions), and their `ec-v3` judge (Opus 5.5) scores the answer as one readout against the
ported 6-way option lists (gold = Qwen's own most frequent reason, 4 cross-question distractors, the
opposite-side reason on 72 committed items, "cannot tell"); deliberative items need both sides in
one answer. `--check vendor/workspace-bench/tests/golden/moral_rationale_options_v2.json` asserts
the lists match theirs item for item. Results in
[docs/wsb-moral-rationale.md](../docs/wsb-moral-rationale.md).

```bash
uv run python -m wsb.moral_rationale --n 3   # all 200 items x 3 draws, ~10 min
```

`user_modeling`, `jailbreak_recognition`, `basic_readout`, `knowledge_cutoff`, `wrong_bridge` and `moral_rationale` so far.
